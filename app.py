import streamlit as st
import logging, traceback
logging.getLogger().setLevel(logging.ERROR)

import re
import os
import json
import mimetypes
import pandas as pd
from glob import glob
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable

import nest_asyncio
nest_asyncio.apply()
import pymupdf
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, ServiceContext, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.query_engine import CustomQueryEngine, SimpleMultiModalQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response

# view source
from llama_index.core.response.notebook_utils import display_source_node
from PIL import Image
import matplotlib.pyplot as plt



FILE_PATH = "./cache/data"
LLAMAPARSER_CHARTS_PATH = "./cache/data/charts"
LLAMAPARSER_IMAGES_PATH = "./cache/data/images"
LLAMAPARSER_OUTPUT_DICT_PATH = "./cache/data/results"
os.makedirs(LLAMAPARSER_OUTPUT_DICT_PATH, exist_ok=True)
BOOL_SPLIT_FILE = True

load_dotenv()

# llm = OpenAI(model="gpt-4o", temperature=0.0, request_timeout=600, max_retries=3)
# # llm = OpenAIMultiModal(model="gpt-4o", max_new_tokens=4096)
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Settings.llm = llm


LLAMAPARSER_INSTRUCTION ="""You are parsing a brief of "Clarkson's Shipping Intelligence Weekly" Report. It contains many complex tables. The instructions below focus on comprehensively understanding and processing images, tables, and text within the document. Special emphasis is placed on accurately identifying the structure of intricate tables to distinguish and convey multilayered, nested information. Follow these instructions carefully and reconstruct contents in a cohesive way. The output should be in **Markdown format without any additional explanations**.

1. If a table's title contains "Voyage," **always must include the first column’s numbers**. Place them in a single field, separated by a bar(”|”).(Exclude if absent).
  - Incorrect : “| 23 270,000t MEG - China* |”
  - Correct : “| 23 | 270,000t MEG - China* |”
2. To distinguish headers from data fields, **insert new separator rows matching the header length**. This row visually separates the header and data sections within the table.
3. If there are **row headers** with 1 or more levels, depending on font size and boldness, **NEVER omit** **any row header groups.** If a header name is empty or undefined, leave the corresponding cell **blank**.
4. If the **column headers** include "Trends" or "This Week," these columns must be included, and their values should be **merged into a single field** without separating words and numbers.
  - Incorrect: ”| FIRM.... | 19% |”
  - Correct: ”| FIRM.... 19% |”
5. All **Bold, italics, symbols** are crucial and must be **preserved** to reflect visually emphasized content and table. Particularly, row headers in tables are in bold, so make sure to be represented strikely.
6. Include **annotations or notes** below the table under the `{table_footer}` section. These notes should provide additional explanations or context related to the table.
7. For **empty data fields**, handle them as empty strings while ensuring the cell position reflects its contextual meaning. If necessary, include clarifications in the notes.
8. Each field should clearly represent the intended meaning of the data and must be accurately extracted considering the data type (e.g., numeric, textual)."""

QA_PROMPT_TMPL = """You are an AI assistant with expertise in shipping business metrics.
You will be given information in 'markdown' mode from PDFs that may include text, tables, and charts related to weekly shipping business performance and industry trends.
Your task is to analyze this information and provide a clear, concise answer to the user's question.
Use the image information first and foremost. ONLY use the text/markdown information if you can't understand the image.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge, answer the query.
Focus on the most relevant data points and insights that directly address the user's query.
Explain whether you got the answer from the parsed markdown or raw text or image, and if there's discrepancies, and your reasoning for the final answer.

Query: {query_str}
Answer: """

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)



class MultimodalQueryEngine(CustomQueryEngine):
    """Custom multimodal Query Engine.

    Takes in a retriever to retrieve a set of document nodes.
    Also takes in a prompt template and multimodal model.

    """

    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: OpenAIMultiModal

    def __init__(self, qa_prompt: Optional[PromptTemplate] = None, **kwargs) -> None:
        """Initialize."""
        super().__init__(qa_prompt=qa_prompt or QA_PROMPT, **kwargs)

    def custom_query(self, query_str: str):
        # retrieve text nodes
        nodes = self.retriever.retrieve(query_str)

        # create ImageNode items from text nodes
        image_nodes = [
            NodeWithScore(node=ImageNode(image_path=n.metadata["page_screenshot_path"]))
            for n in nodes
        ]

        # create context string from text nodes, dump into the prompt
        context_str = "\n\n".join(
            [r.get_content(metadata_mode=MetadataMode.LLM) for r in nodes]
        )
        fmt_prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)

        # synthesize an answer from formatted text and images
        llm_response = self.multi_modal_llm.complete(
            prompt=fmt_prompt,
            image_documents=[image_node.node for image_node in image_nodes],
        )
        return Response(
            response=str(llm_response),
            source_nodes=nodes,
            metadata={"text_nodes": nodes, "image_nodes": image_nodes},
        )

        return response


def fn_split_pdf(filepath, newfilepath=None, batch_size=None):
    # inspired by 'teddynote'
    """
    PDF를 여러 개의 작은 pdf 파일로 분할
    """
    if newfilepath is None:
        newfilepath = filepath

    input_pdf = pymupdf.open(filepath)
    num_pages = len(input_pdf)
    print("Total page count : ", num_pages)

    if batch_size is None:
        batch_size = num_pages # step을 전체 페이지로 설정함, 즉 실제론 split하지 않음
    split_output_files=[]
    # pdf 분할
    for start_page in range(0, num_pages, batch_size):
        end_page = min(start_page+batch_size, num_pages) - 1

        # save the split pdf
        # input_file_basename = os.path.splitext(filepath)[0]
        input_file_basename = os.path.basename(filepath) #
        input_file_basename = os.path.splitext(input_file_basename)[0] # 파일 타입 제거
        print("input_file_basename :", input_file_basename)
        output_file = f"{input_file_basename}_{start_page:04d}_{end_page:04d}.pdf"
        output_file = os.path.join(newfilepath, output_file)
        print("분할 pdf 생성 : ", output_file)
        # new empty PDF
        with pymupdf.open() as output_pdf:
            # merge
            output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
            output_pdf.save(output_file)
            split_output_files.append(output_file)
    # 입력 pdf 닫기
    input_pdf.close()
    print("...Complete...")
    return split_output_files



def _get_clarksons_report_publish_datetime(file_path):
    """
    Extract metadata from a Clarksons report file name.

    Args:
        file_path (str): File name including the date in MM_DD_YYYY format.

    Returns:
        dict: Metadata containing 'publish_date' in 'YYYY-MM-DD' format or an error message.
    """
    # 날짜 추출 정규식
    date_pattern = r"(\d{2})_(\d{2})_(\d{4})"
    match = re.search(date_pattern, os.path.basename(file_path))

    if match:
        # 날짜 구성 요소 추출
        # month, day, year = match.groups()
        day, month, year = match.groups()
        # 날짜 객체로 변환 후 형식 변경
        formatted_date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%Y-%m-%d")
        # return {"publish_date": formatted_date}
        return formatted_date
    else:
        # return {"error": "Unable to extract date from the file name"}
        return "Unable to extract date from the file name"

def get_structured_json_dicts(input_json_dicts):

    full_page_result = []
    for element in input_json_dicts:
        """
        page_elements : dict[List[dict]] # header, table etc
        page_metadata : dict[int, dict]
        page_summary : dict[int, str] # 추후

        page_screenshot: list[str] # page screenshot paths
        page_screenshot_summmary : list[str] # image summary # 추후

        page_chart : list[dict] # chart/table infos
        page_chart_summary : list[str] # 추후

        page_text : list[str] # markdown text
        page_text_summary: list[str] # 추후
        """

        page_elements = element['items']
        page_screenshot = [
            item for item in element['images'] if 'type' in item and item['type'] == 'full_page_screenshot'
        ]
        page_chart = [
            item for item in element['charts']
            if re.match(r'chart_p(\d+)_(?!0)(\d+)\.png', item['name'])
        ]

        # for save
        per_page_dict = {
          'markdown_text': element['md'],
          'page_num' : element['page'],
          'page_screenshot_info':page_screenshot,
          'page_table_info':page_chart,
          'page_elements':page_elements,
        }
        full_page_result.append(per_page_dict)
    assert len(input_json_dicts) == len(full_page_result)

    return full_page_result

def plot_single_img(img_path):
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()

def main_llamaparser():

    # 1. split pdf
    file_path_list = glob(os.path.join(FILE_PATH, "*.pdf"))
    if BOOL_SPLIT_FILE :
        # 하위 디렉토리에 'split' 폴더를 새롭게 만든 후 fn_split_pdf 함수를 실행한다.
        # './clarksons-ShippingIntelligenceWeekly/split'
        SPLIT_FILE_PATH = os.path.join(FILE_PATH, "split")
        print("SPLIT_FILE_PATH :", SPLIT_FILE_PATH)
        os.makedirs(SPLIT_FILE_PATH, exist_ok=True)
        # file_path_list = fn_split_pdf(file_path_list[-1], SPLIT_FILE_PATH, batch_size=10)
        # while True:
        new_file_path_list = []
        for file in file_path_list[:2]:
            new_file_path_list.extend(fn_split_pdf(file, SPLIT_FILE_PATH))

    # 2. create llamaparse
    parser = LlamaParse(
          # leverages state-of-the-art multimodal models
          premium_mode=True,

          # text and images handing
          result_type="markdown",
          language="en",
          skip_diagonal_text=True, # (default True)
          do_not_unroll_columns=True,
          disable_image_extraction=False,
          extract_charts=True, # save
          page_separator="\n\n====\n\n",

          # parsing instruction
          parsing_instruction=LLAMAPARSER_INSTRUCTION,
          is_formatting_instruction=True,

          # cache
          invalidate_cache=True,# keeps results cached for 48 hours after upload.
      )
    
    # 3. start parse
    with open(os.path.join(LLAMAPARSER_OUTPUT_DICT_PATH, 'full_parsed_results.jsonl'), "w") as out_file:
        for file_path in new_file_path_list: 
            # print("file_path :", file_path)
            if os.path.isfile(file_path):
                # start parsing
                md_json_objs = parser.get_json_result(file_path)
                # print("...done parse...")

                # extract markdown data
                json_dicts = md_json_objs[0]["pages"]
                # print("...done extract markdown data...")

                # change dir
                file_name = os.path.basename(file_path).split(".pdf")[0]

                # extract structured data from 'json_dicts'
                structured_json_dicts = get_structured_json_dicts(json_dicts)
                # [todo] page 별 element 데이터 parquet로 저장하기
                # sub_df = pd.DataFrame([-])
                # sub_df.to_parquet(os.path.join(LLAMAPARSER_OUTPUT_DICT_PATH, f"parsed_{file_name}.parquet"), index=False)

                # extract charts & images(screenshot) from PDF and save them
                # !@! 이 코드 실행해야 'md_json_objs'안에 이미지/차트 경로가 추가됨
                # extract charts
                chart_dir = os.path.join(LLAMAPARSER_CHARTS_PATH, file_name)
                os.makedirs(chart_dir, exist_ok=True)
                # get the SDK to download all the images to a local directory for us
                _ = parser.get_charts(md_json_objs,download_path=chart_dir)
                # print("...done extract charts...")
                # extract images(screenshot)
                image_dir = os.path.join(LLAMAPARSER_IMAGES_PATH, file_name)
                os.makedirs(image_dir, exist_ok=True)
                _ = parser.get_images(md_json_objs,download_path=image_dir)
                # print("...done extract images...")

                # save_dict
                sub_dict = {
                    "json_dicts" : structured_json_dicts,
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": mimetypes.guess_type(file_path)[0],
                    "file_size": os.path.getsize(file_path),
                    "file_publish_datetime": _get_clarksons_report_publish_datetime(file_name),
                    "chart_dir" : chart_dir,
                    "image_dir" : image_dir,
                    "creation_datetime": datetime.fromtimestamp(
                        Path(file_path).stat().st_ctime
                      ).strftime("%Y-%m-%d"),
                    "last_modified_datetime": datetime.fromtimestamp(
                        Path(file_path).stat().st_mtime
                      ).strftime("%Y-%m-%d"),
                    "last_accessed_datetime": datetime.fromtimestamp(
                        Path(file_path).stat().st_atime
                      ).strftime("%Y-%m-%d"),

                }
                out_file.write(json.dumps(sub_dict) + "\n")
                # print(f"...done save file_dict...{file_name}")
                # print()


def main_streamlit():
    # Set my OpenAI API key from the app's secrets.
    # openai_api_key = st.secrets["openai_api_key"]
#     if openai_api_key:

#         # OpenAI LLM 모델 설정
#         llm=OpenAIMultiModal(
#             model=selected_model, 
#             temperature=0, 
#             max_new_tokens=4096,
#             api_key=openai_api_key
#           )
#         Settings.llm = llm

#         # Embeded Model 설정
#         Settings.embed_model = OpenAIEmbedding(
#             mode="similarity", model="text-embedding-3-small", api_key=openai_api_key
#         )

    # Add a heading for app
    st.set_page_config(page_title="Marina ChatBoat", page_icon=":ship:", layout='wide')
    # st.set_page_config(page_title="Marina ChatBoat", page_icon='path/to/your_icon.png', layout='wide')

    st.title(" How can I help you today? ")
    # st.header("Select a prompt below or write your own to start chatting with MarinaGPT.")
    st.info("Select a prompt below or write your own to start chatting with MarinaGPT", icon="📃")


    if "messages" not in st.session_state.keys(): # Initialize the chat message history

        # 첫대화 # # 대화기록을 저장하기 위한 용도로 생성한다.
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Ask me a question!"
            }
        ]

    # 캐시 디렉토리 생성
    if not os.path.exists("./cache"):
        os.mkdir("./cache")

    # 파일 업로드 전용 폴더
    cache_data_dir = "./cache/data"
    if not os.path.exists(cache_data_dir):
        os.mkdir(cache_data_dir)

    # 초기화 버튼 클릭 시 실행
    def _st_clear_chat():
        try:
            st.session_state["messages"] = []
            if "chat_engine" in st.session_state:
                del st.session_state["chat_engine"]

            if "show_warning" in st.session_state:
                del st.session_state["show_warning"]

            # cache/data 폴더에 업로드된 파일 삭제
            if os.path.exists(cache_data_dir):
                for file_name in os.listdir(cache_data_dir):
                    file_path = os.path.join(cache_data_dir, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            st.session_state["chatbot_api_key"] = ""

        except Exception as e:
            st.error(f"An error occurred while resetting: {e}")

    # 사이드바 생성
    with st.sidebar:
        # 초기화 버튼 생성
        clear_btn = st.button("Reset Chat History", on_click=_st_clear_chat)

        # LLM 선택
        selected_model = st.selectbox(
            "Select Model", ("gpt-4o","gpt-4o-mini"), index=0
        )

        # OpneAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key", key="chatbot_api_key", type="password"
        )

        # 파일 업로드
        uploaded_files = st.file_uploader(
            "Upload your file",
            # type=["txt", "pdf", "md", "ppt", "doc", "hwp", "csv", "pptx"],
            type=['pdf'],
            accept_multiple_files=True,
            key="uploaded_files",
        )

        # # 파일 업로드 실행
        # process = st.button("Process")

    if openai_api_key:
        # OpenAI LLM 모델 설정
        llm=OpenAI(
            model=selected_model, 
            temperature=0, 
            # max_new_tokens=4096,
            request_timeout=600, 
            max_retries=3,
            api_key=openai_api_key
        )

        Settings.llm = llm

        # Embeded Model 설정
        embedding_model = OpenAIEmbedding(
            mode="similarity", model="text-embedding-3-small", api_key=openai_api_key
        )
        Settings.embed_model = embedding_model

    else:
        st.warning("Please add your OpenAI API key to continue.")
        st.stop() 

    # 이전 대화를 출력
    def _st_print_messages():
        # for chat_message in st.session_state["messages"]:
        #     # print(chat_message)
        #     st.chat_message(chat_message["role"]).write(chat_message["content"])
        for chat_message in st.session_state["messages"]: # Display the prior chat messages
            with st.chat_message(chat_message["role"]):
                # st.write(chat_message["content"])
                st.markdown(chat_message["content"])


    # 새로운 메시지를 추가
    def _st_add_to_message(role, content):
        message = {"role": role, "content": str(content)}
        st.session_state["messages"].append(message)  # Add response to message history

    # 업로드 파일을 캐시 폴더에 저장
    @st.cache_resource(show_spinner=True)
    def _st_load_index_data(file):
        try:
            with st.spinner(text="Loading and indexing the uploaded docs – This should take 1-2 minutes."):
                NODES_DIR = os.path.join(cache_data_dir, "parser_preminum_2_nodes.pkl")
                # 업로드한 파일을 캐시 디렉토리에 저장.
                # file_content = file.read()
                # file_path = f"./cache/data/{file.name}"
                # with open(file_path, "wb") as f:
                #     f.write(file.getvalue())
                import pickle
                nodes = pickle.load(open(NODES_DIR, "rb"))
                index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
                index.storage_context.persist(persist_dir=PERSIST_DIR)
                return index

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            logging.error(f"File processing error: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def _st_get_jpg_name(path):
        match = re.search(r'page_\d+\.jpg$', path)
        if match:
            return match.group()
        return None

    # 이전 대화 기록 출력
    _st_print_messages()

    # 경고 메시지를 띄우기 위한 빈 영역
    warning_msg = st.empty()

    if uploaded_files is not None:
        try:
            
            PERSIST_DIR = os.path.join("./cache/store", 'storage_clarksons_nodes')
            if not os.path.exists(PERSIST_DIR):
                index = _st_load_index_data(uploaded_files)
                # import pickle
                # nodes = pickle.load(open(NODES_DIR, "rb"))
                # index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
                # index.storage_context.persist(persist_dir=PERSIST_DIR)

            else:

                ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                index = load_index_from_storage(ctx)

            # # 3. query engine 불러오기 # 안댐
            # index = MultimodalQueryEngine(
            #     retriever=index.as_retriever(similarity_top_k=3), multi_modal_llm=llm
            # )
            print("...check index:", index)
            # if "chat_engine" not in st.session_state:  # Initialize the query engine
                # st.session_state["chat_engine"] = index.as_chat_engine(
                #     chat_mode="condense_question", verbose=True, streaming=True
                # ) # 'condense_question' 'reAct_agent' 'OpenAI_agent'

            if "conversation" not in st.session_state: # test
                st.session_state["conversation"] = MultimodalQueryEngine(
        retriever=index.as_retriever(similarity_top_k=3), multi_modal_llm=llm
    )
        except Exception as e:
            st.error(f"An error occurred while generating the index: {str(e)}")
            logging.error(f"Index generation error: {str(e)}")
            logging.error(traceback.format_exc())

    else:
        if "show_warning" not in st.session_state:
            st.session_state["show_warning"] = True  # 경고 메시지 표시 플래그
        if st.session_state["show_warning"]:
            st.warning("Please upload a file.")


    # chat logic
    # 사용자 입력을 받아서 처리하는 부분.
    if query:= st.chat_input("Ask a question..."):
        _st_add_to_message("user", query)
        # _st_print_messages() # 확인!!

        # 사용자의 입력을 출력함.
        # st.chat_message("user").write(query)
        with st.chat_message("user"):
            st.markdown(query)

        # If last message is not from assistant, generate a new response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                chain=st.session_state["conversation"]
                print("...check chain :", chain)
                try:
                    with st.spinner("Thinking..."):
                        response = chain.query(query)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        # print("...check chain :", response)
                        # print("...check response source :", response.source_nodes)
                        # print("...check response source :", response.metadata)
                        # for text_node in response.source_nodes:
                            # print("relevant source :", os.path.basename(text_node.metadata["original_file_path"]))
                            # print("relevant page number :", text_node.metadata['page_num'])
                            # print("relevant page screen :", text_node.metadata['page_screenshot_path'])
                            # page_screenshot_path = text_node.metadata['page_screenshot_path']
                            # st.image(page_screenshot_path, caption=_st_get_jpg_name(page_screenshot_path))
                            # display_source_node(text_node, source_length=1000)

                        with st.expander("view references - sources, page numbers, and screenshots"):
                            for text_node in response.source_nodes:
                                page_screenshot_path = text_node.metadata['page_screenshot_path']
                                page_screenshot_info = text_node.metadata['page_screenshot_info']
                                page_screenshot_info = [
                                        item['original_file_path'] for item in page_screenshot_info if 'type' in item and item['type'] == 'full_page_screenshot'
                                    ][0]
                                relevant_file_path = os.path.basename(page_screenshot_info)
                                relevant_page_num = text_node.metadata['page_num']
                                relevant_source = f"Page Number {relevant_page_num} of {relevant_file_path}"
                                # st.markdown(relevant_source, help = display_source_node(text_node.text, source_length=1000))
                                st.markdown(relevant_source, help = text_node.text)
                                # st.markdown(relevant_source, help = text_node) # 안댐
                                st.image(page_screenshot_path, caption=_st_get_jpg_name(page_screenshot_path))
                                # st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                                # st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                                

                        # response_stream = st.session_state["chat_engine"].chat(query)
                        # # st.write_stream(response_stream.response_gen) # error
                        # st.write(response_stream.response)
                        # message = {"role": "assistant", "content": response_stream.response}
                        # # Add response to message history
                        # st.session_state.messages.append(message)


                except Exception as e:
                    response_container = st.empty()
                    response_str = f"An error occurred. Please ensure the uploaded file is available.\n [error] {e}"
                    response_container.markdown(response_str.replace("\n", "  \n"))
                    logging.error(f"Response generation error: {str(e)}")
                    logging.error(traceback.format_exc())
                    st.session_state["show_warning"] = False


        # # 스트리밍 호출
        # with st.chat_message("assistant"):
        #     response_str = ""
        #     response_container = st.empty()

        #     try:
        #         # container에 토큰을 스트리밍 출력
        #         with st.spinner("Thinking..."):
        #             response = st.session_state["chat_engine"].stream_chat(query)

        #             for token in response.response_gen:
        #                 response_str += token
        #                 response_container.markdown(response_str)

        #         # source_documents = result['source_documents']
        #         # with st.expander("참고 문서 확인"):
        #         #     st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
        #         #     st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
        #         #     st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
        #     except Exception as e:
        #         response_str = f"An error occurred. Please ensure the uploaded file is available.\n [error] {e}"
        #         response_container.markdown(response_str.replace("\n", "  \n"))
        #         logging.error(f"Response generation error: {str(e)}")
        #         logging.error(traceback.format_exc())
        #         st.session_state["show_warning"] = False
            
        # # Add assistant message to chat history
        # _st_add_to_message("assistant", response_str)


if __name__ == "__main__":
    main_streamlit()
    