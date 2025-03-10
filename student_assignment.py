import datetime
import chromadb
import traceback

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

import csv

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

db_path = "./"
csv_path = "./COA_OpenData.csv"

def generate_hw01():
    try:
        # 初始化Chroma客戶端
        chroma_client = chromadb.PersistentClient(path=db_path)
        
        # 設置OpenAI嵌入函數
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=gpt_emb_config['api_key'],
            api_base=gpt_emb_config['api_base'],
            api_type=gpt_emb_config['openai_type'],
            api_version=gpt_emb_config['api_version'],
            deployment_id=gpt_emb_config['deployment_name']
        )
        
        # 創建/獲取Collection（嚴格匹配參數）
        collection = chroma_client.get_or_create_collection(
            name="TRAVEL",
            metadata={"hnsw:space": "cosine"},
            embedding_function=openai_ef
        )
        
        # 讀取並處理CSV數據
        with open(csv_path, 'r', encoding='utf-8-sig') as file:  # 處理BOM標記
            csv_reader = csv.DictReader(file)
            
            batch_size = 100
            documents, metadatas, ids = [], [], []
            
            for i, row in enumerate(csv_reader):
                # 時間戳轉換（含錯誤處理）
                try:
                    date_obj = datetime.datetime.strptime(
                        row['CreateDate'], 
                        "%Y-%m-%d"  # 根據實際格式調整
                    )
                    timestamp = int(date_obj.timestamp())
                except Exception as e:
                    print(f"行 {i+1} 日期解析失敗: {str(e)}")
                    timestamp = 0  # 設置默認值

                # 構建metadata
                metadata = {
                    "file_name": csv_path.split("/")[-1],
                    "name": row['Name'],
                    "type": row['Type'],
                    "address": row['Address'],
                    "tel": row['Tel'],
                    "city": row['City'],
                    "town": row['Town'],
                    "date": timestamp
                }
                
                # 填充數據批次
                documents.append(row['HostWords'])  # 核心文本字段
                metadatas.append(metadata)
                ids.append(f"travel_{i}")

                # 批次提交
                if (i+1) % batch_size == 0:
                    collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    documents, metadatas, ids = [], [], []
            
            # 提交殘留數據
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )

        print(f"數據入庫完成，共處理 {i+1} 條記錄")
        print(f"元數據範例：{metadata}")  # 打印最後一筆metadata驗證

        return collection
    except KeyError as ke:
        print(f"CSV欄位缺失：{str(ke)}，請檢查CSV結構")
    except Exception as e:
        print(f"致命錯誤：{str(e)}")
        traceback.print_exc()
    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=db_path)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection
