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
        if collection.count() == 0:

            # 在循環開始前初始化
            metadata = None

            # CSV数据处理(带格式验证)
            batch_docs, batch_metas, batch_ids = [], [], []
            with open(csv_path, 'r', encoding='utf-8-sig') as file:
                csv_reader = csv.DictReader(file)

                for i, row in enumerate(csv_reader):

                    # 元数据构建
                    metadata = {
                        "file_name": csv_path,
                        "name": row['Name'],
                        "type": row['Type'],
                        "address": row['Address'],
                        "tel": row.get('Tel', ''),  # 处理可选字段
                        "city": row['City'],
                        "town": row['Town'],
                        "date": datetime.datetime.strptime(row["CreateDate"], '%Y-%m-%d').timestamp()
                    }

                    batch_docs.append(row['HostWords'])
                    batch_metas.append(metadata)
                    batch_ids.append(str(i))

            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )

            # 调试信息
            if debug:
                print(f"共處理 {i+1} 條數據 | 數據庫路徑: {os.path.abspath(db_path)}")
                print(f"示例元數據: {metadata}")
                print(f"集合統計: {collection.count()} 條記錄")

        else:
            if debug:
                print("本次未新增任何數據")
                print(f"集合統計: {collection.count()} 條記錄")
        return collection

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"數據處理失敗: {str(e)}") from e

def generate_hw02(question, city, store_type, start_date, end_date):
    collection = generate_hw01()
    # 构建查询条件
    where_conditions = {"$and": []}
    where_conditions["$and"].append(
            {"city": {"$in": city}}
        )
    where_conditions["$and"].append(
            {"type": {"$in": store_type}}  # 確保使用列表值
        )
    where_conditions["$and"].append(
            {"date": {"$gte": int(start_date.timestamp())}}
        )
    where_conditions["$and"].append(
            {"date": {"$lte": int(end_date.timestamp())}}
        )
    print(where_conditions)

    # 执行查询
    results = collection.query(
        query_texts=[question],
        n_results=10,
        where=where_conditions if where_conditions else None,
        include=["metadatas", "distances"]
    )

    # 处理结果
    filtered_results = []
    for distance, metadata in zip(results['distances'][0], results['metadatas'][0]):
        similarity = 1 - distance  # 余弦相似度转换
        if similarity >= 0.80:
            filtered_results.append(metadata['name'])
    
    return filtered_results[:10]

def generate_hw03(question, store_name, new_store_name, city, store_type):
    collection = generate_hw01()
    # 更新指定店家的元数据
    try:
        # 查找目標店家
        target = collection.get(
            where={"name": {"$eq": store_name}},
            include=["metadatas"]
        )
        
        # 更新元数据
        if target['ids']:
            new_metadata = target['metadatas'][0].copy()
            new_metadata["new_store_name"] = new_store_name
            collection.update(
                ids=target['ids'][0],
                metadatas=new_metadata
            )
    except Exception as e:
        print(f"更新失敗: {str(e)}")
    
    # 構建查詢條件
    where_conditions = {"$and": []}
    if city:
        where_conditions["$and"].append({"city": {"$in": city}})
    if store_type:
        where_conditions["$and"].append({"type": {"$in": store_type}})
    
    # 執行語意查詢
    results = collection.query(
        query_texts=[question],
        n_results=10,
        where=where_conditions if where_conditions["$and"] else None,
        include=["metadatas", "distances"]
    )
    
    # 處理結果並過濾
    filtered = []
    for distance, meta in zip(results['distances'][0], results['metadatas'][0]):
        similarity = 1 - distance
        if similarity >= 0.80:
            display_name = meta.get("new_store_name", meta['name'])  # 優先顯示新名稱
            filtered.append({
                "name": display_name,
                "score": round(similarity, 3),
                "original_name": meta['name']
            })
    
    # 排序與去重
    sorted_results = sorted(filtered, key=lambda x: x['score'], reverse=True)
    seen = set()
    final = [x['name'] for x in sorted_results 
             if not (x['original_name'] in seen or seen.add(x['original_name']))]
    
    return final[:10]  # 確保最多返回10個

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
