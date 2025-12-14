import chromadb # type: ignore
from chromadb.config import Settings # type: ignore
from sentence_transformers import SentenceTransformer
import os
import json

class RegionDB:
    def __init__(self, db_path=None, collection_name="region_cache"):
        if db_path is None:
            # 获取 db_manager.py 所在的绝对目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 将数据库固定存放在 Material_Library/Database/chroma_db
            db_path = os.path.join(current_dir, "chroma_db")
        # 初始化 ChromaDB (持久化存储)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # 初始化 Embedding 模型 (用于将 Prompt 转为向量)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✅ ChromaDB Initialized at {db_path}")

    def add_region(self, prompt, region_name, file_path, indices_list):
        """
        在 Capture 阶段调用：存入 Prompt, Region Name, 和物理路径
        """
        embedding = self.encoder.encode(prompt)
        
        indices_str = json.dumps(indices_list) if isinstance(indices_list, list) else str(indices_list)

        self.collection.upsert(
            ids=[file_path],  # 使用绝对路径作为唯一 ID
            embeddings=[embedding.tolist()],
            metadatas=[{
                "prompt": prompt,
                "region_name": region_name,
                "file_path": file_path,
            }]
        )

    def search_region(self, query_prompt, n_results=1):
        """
        在 Inference 阶段调用：根据 Prompt 查找最相似的 Region
        """
        query_vec = self.encoder.encode(query_prompt).tolist()
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results
        )
        
        # 解析返回结果
        found_items = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                item = {
                    "id": results['ids'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                }
                found_items.append(item)
        return found_items
