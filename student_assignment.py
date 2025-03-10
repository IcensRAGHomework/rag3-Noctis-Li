import os
import datetime
import chromadb
import traceback

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

import csv

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

db_path = "./"
csv_path = "COA_OpenData.csv"

def generate_hw01(debug=False):
    try:
        # 初始化Chroma客戶端
        settings = chromadb.config.Settings(persist_directory="chroma.sqlite3")
        chroma_client = chromadb.PersistentClient(path=db_path, settings=settings)

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

        # 增量更新检查
        if collection.count() > 0:
            print("集合已存在数据，启用增量更新模式") if debug else None
            existing_ids = set(collection.get()['ids'])
        else:
            existing_ids = set()

        # 在循環開始前初始化
        metadata = None
        data_processed = False

        # CSV数据处理(带格式验证)
        with open(csv_path, 'r', encoding='utf-8-sig') as file:
            csv_reader = csv.DictReader(file)
            
            # 验证CSV结构
            required_fields = {'HostWords', 'Name', 'Type', 'Address', 'Tel', 
                              'City', 'Town', 'CreateDate'}
            if not required_fields.issubset(csv_reader.fieldnames):
                missing = required_fields - set(csv_reader.fieldnames)
                raise KeyError(f"CSV缺少必要字段: {missing}")

            # 批次处理优化
            batch_size = 500  # 根据API速率限制调整
            counter = 0
            for i, row in enumerate(csv_reader):
                # 跳过已存在数据
                doc_id = f"travel_{i}"
                if doc_id in existing_ids:
                    continue

                # 日期解析
                try:
                    date_obj = datetime.datetime.strptime(
                        row['CreateDate'], 
                        "%Y-%m-%d"  # 根據實際格式調整
                    )
                    timestamp = int(date_obj.timestamp())
                except Exception as e:
                    print(f"行 {i+1} 日期解析失敗: {str(e)}")
                    timestamp = 0  # 設置默認值

                # 元数据构建
                metadata = {
                    "file_name": os.path.basename(csv_path),
                    "name": row['Name'],
                    "type": row['Type'],
                    "address": row['Address'],
                    "tel": row.get('Tel', ''),  # 处理可选字段
                    "city": row['City'],
                    "town": row['Town'],
                    "date": timestamp
                }
                data_processed = True

                # 批次提交
                if counter == 0:
                    batch_docs, batch_metas, batch_ids = [], [], []
                
                batch_docs.append(row['HostWords'])
                batch_metas.append(metadata)
                batch_ids.append(doc_id)
                counter += 1

                if counter >= batch_size:
                    collection.add(
                        documents=batch_docs,
                        metadatas=batch_metas,
                        ids=batch_ids
                    )
                    counter = 0
                    if debug:
                        print(f"已提交批次: {i//batch_size + 1}")

            # 提交残留批次
            if counter > 0:
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )

        # 调试信息
        if debug:
            print(f"共處理 {i+1} 條數據 | 數據庫路徑: {os.path.abspath(db_path)}")
            if data_processed:
                print(f"示例元數據: {metadata}")
            else:
                print("本次未新增任何數據")
            print(f"集合統計: {collection.count()} 條記錄")

        return collection

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"數據處理失敗: {str(e)}") from e
    
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
