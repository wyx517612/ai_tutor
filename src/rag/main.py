# main.py
import os
from src .rag .splitter import split_into_chunks
from src .rag .embedder import embed_chunk
from src .rag .retriever import save_embeddings, retrieve, rerank
from src .rag .quiz_generator import Generator

# 配置
DOC_PATH = "data/第一章 绪论.txt"  # 改成你的数据结构教材txt路径
TOP_K = 5
RERANK_TOP_K = 3


def main():
    print("🚀 数据结构AI助教启动中...")

    # 1. 检查环境变量
    if not os.environ.get('DEEPSEEK_API_KEY'):
        print("⚠️ 未设置 DEEPSEEK_API_KEY 环境变量")
        print("请运行: set DEEPSEEK_API_KEY=你的API密钥")
        return

    # 2. 加载并分割
    print("📖 加载文档...")
    chunks = split_into_chunks(DOC_PATH)
    print(f"✅ 分割为 {len(chunks)} 个块")

    # 3. 向量化
    print("🧮 生成向量...")
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    print(f"✅ 生成了 {len(embeddings)} 个向量")

    # 4. 保存到向量数据库
    print("💾 保存到向量数据库...")
    save_embeddings(chunks, embeddings)

    # 5. 初始化生成器
    generator = Generator()

    # 6. 问答循环
    print("\n✅ 准备就绪！输入问题开始提问（输入 'quit' 退出）\n")

    while True:
        query = input("👨‍🎓 你：")
        if query.lower() == 'quit':
            print("👋 再见！")
            break

        if not query.strip():
            print("⚠️ 请输入有效问题")
            continue

        # 检索
        print("🔍 检索中...")
        retrieved = retrieve(query, TOP_K)
        reranked = rerank(query, retrieved, RERANK_TOP_K)

        # 生成回答
        print("🤖 生成回答中...")
        answer = generator.generate(query, reranked)

        print(f"\n🧑‍🏫 助教：{answer}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()