![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/Screenshot_8.png)
<br><br>

# **Introduction**
the solution is the RAG System for personal career advice based on sampled Wuzzuf job postings utilizing SOTA approaches for data indexing, retrieval, and Quality Generation for advice 

# **Environment, Frameworks and Deployment**
1. Kaggle for Notebook hosting using its GPU Accelerator <br>
2. Ollama was utilized for speeding up models' performance
3. LangChain was the framework used for building RAG Chains
4. LLama3, Newly Released Gemma2 9b on Ollama was used
5. Evaluation was done using LangSmith Evaluators and RAGAS based on the OpenAI GPTs model was used for evaluation

# **Installing Dependencies**
either you follow the installation on the notebook<br>
or you can install it if you are working on a local machine using 
```console

pip install -r requirements.txt
```
install Ollama and pull models
```console

curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llama3 #Pulls LLama3 8B
ollama pull gemma2 #Pulls New model Gemma2 9b

```
# **LangChain Framework**
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/LangChain.png)
<br> <br> 
I am utilizing the [LangChain](https://www.langchain.com/) Framework for building RAG Chain taking advantage of its ease of use and stable performance 

# **LLM-Embedder**
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/llm-embedder.png)
<br> <br> 
LLMs face a series of challenges, including<br>
issues such as hallucination, instruction following <br>
and handling long contexts. Many of these challenges can be
traced back to the inherent limitations of LLMs, with three critical
boundaries deserving attention.<br>
• **Knowledge boundary**. LLMs are constrained by their knowledge capacity. Due to finite model parameters, they cannot fully
internalize the vast body of world knowledge. Moreover, the internal knowledge of LLMs is static and difficult to be updated with
the dynamically evolving world. Furthermore, LLMs are predominantly trained on publicly available, high-frequency data, which
may result in inaccuracies when dealing with domain-specific or
long-tail knowledge.<br>
• **Memory boundary**. LLMs also grapple with severe limitations
in memory, primarily due to restrictions on context length. While
advances have been continually made in expanding the maximum
context length, it still falls short of achieving the goal of lifelong
engagement with human users. Additionally, both the training and
deployment of LLMs with extended context can be prohibitively
computationally and storage-intensive, making it impractical to
significantly expand their memory.<br>
• **Capability boundary**. LLMs’ capabilities are constrained
in terms of action and autonomy. Firstly, they are limited to the
’language space’ and cannot meaningfully interact with the physical
world. Secondly, these models heavily rely on human guidance,
requiring clear user instructions and appropriate demonstration
examples to perform specific tasks effectively.<br>

The above inherent boundaries cannot be effectively addressed by LLMs alone. <br>
To overcome these limitations, external assistance is sought through the process known as **retrieval-augmented generation**. <br>
Retrievers play a crucial role in connecting LLMs with the necessary external components, enabling LLMs to accomplish various downstream tasks. <br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/Screenshot_9.png)

In this context, several common types of retrievers have been designed,<br>
each tailored to fulfill a distinct role in enhancing LLMs:<br>
• **Knowledge Retriever**: providing external knowledge to support LLMs in tackling knowledge-intensive tasks.<br>
• **Memory Retriever**: collecting information that extends beyond the immediate context, assisting in the generation of lengthy
sequences.<br>
• **Tool Retriever**: selecting appropriate tools, allowing LLMs to
interact effectively with the physical world.<br>
• **Example Retriever**: locating pre-cached demonstration examples, from which LLM prompts can be automatically generated
to facilitate in-context learning.<br>

Based on this LLM-Embedder, a unified embedding model to satisfy the primary retrieval augmentation needs of LLMs. Unifying the diverse retrieval capabilities holds significant advantages.
From a practical standpoint, LLM-based systems often require multiple external modules, such as knowledge bases, memory stores,
and tool-bench, to execute complex tasks. By consolidating these functionalities into a unified model, we can streamline system management and enhance operational efficiency. From the perspective
of effect, the unified model may also benefit from the composite data of different scenarios. This can be especially helpful for retrieval tasks where high-quality training data is scarce <br>

Here is how LLM Embedder is loaded to the RAG Chain:<br>
```python
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

model_name = "BAAI/llm-embedder"
embd = HuggingFaceEmbeddings(
    model_name=model_name,
)
```
<br><br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/LLM-Embedder-Comparision.png)


# **Vectorstore Creation and loading using Milvus**
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/998c09ca-cfa6-4c01-ac75-3dfad7f4862b.png)
<br>
**Milvus** stands out as the most comprehensive solution among the databases evaluated, meeting all the essential criteria and outperforming other open-source options<br><br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/VectorDB%20Comparision.png)

<br>

In The Notebook Shared on Google Colab Pro Nivida T4 It takes 30+ mins to create Vectorstore
```python
from langchain_community.document_loaders import CSVLoader
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
docs = CSVLoader("sampled_jobs.csv").load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs)

vectorstore = Milvus.from_documents(  
    documents=doc_splits,
    embedding=embd,

    connection_args={
        "uri": "./milvus_demo.db",
    },
    drop_old=True,  # Drop the old Milvus collection if it exists

)

```
<br><br>

I already shared in the notebook the SampleJobs_DB.db generated VectorStore so no need to recreate it if you are testing my code <br>

```python
#download the vectorestore using gdown
!gdown https://drive.google.com/uc?id=1-8p5GHC8eZCJUU54Zwg-iUW8FasGtatr
#Load the vectorstore
from milvus import default_server
default_server.start() # Start Milvus Server in order to load the store vectorstore

from langchain_milvus import Milvus

vectorstore = Milvus(
    embd,
    connection_args={"uri": "SampleJobs_DB.db"},
)
```

<br>
After Loading we can use Milvus reteriver<br>

```python
Milvus_retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) # k here is the number of retrieved job postings here I use the best single match feel free to exercise with more job postings
```
<br><br>
# **Improving RAG Chain with HyDE**
or **Precise Zero-Shot Dense Retrieval without Relevance Labels**
<br><br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/download%20(4).png)
<br><br>

Dense retrieval is effective across various tasks and languages, but creating efficient fully zero-shot dense retrieval systems without relevance labels is challenging. This paper addresses the difficulty of zero-shot learning and encoding relevance by proposing Hypothetical Document Embeddings (HyDE).<br>

HyDE operates in two steps:

1. It instructs a language model (e.g., InstructGPT) to generate a hypothetical document based on a query. This document captures relevant patterns but may contain false details.<br>
2. An unsupervised contrastively learned encoder (e.g., Contriever) encodes the hypothetical document into an embedding vector, which is used to find similar real documents in the corpus based on vector similarity. This step grounds the generated document in the actual corpus, filtering out incorrect details through the encoder's dense bottleneck.<br>

Experiments show that HyDE significantly outperforms the state-of-the-art unsupervised dense retriever Contriever and achieves performance comparable to fine-tuned retrievers across various tasks such as web search and QA. <br>

![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/HyDE%20Comparison.png)

<br>
How is it implemented in my RAG Chain?<br>

```python
from langchain_community.chat_models import ChatOllama
model = ChatOllama(model='gemma2', temperature=0) #or you can use llama3

from langchain.prompts import ChatPromptTemplate

# HyDE document genration
template = """Please write a scientific paper passage to answer the question
Question: what are the requirements needed to become a good {question}?
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser

generate_docs_for_retrieval = (
    prompt_hyde | model | StrOutputParser()
)
question = "Network Engineer"
HyDE_retrieval_chain = generate_docs_for_retrieval | Milvus_retriever
retireved_docs = HyDE_retrieval_chain.invoke({"question":question})
pretty_print_docs(retireved_docs)

```
<br>
output
<br>

```console
Document 1:

job_title: Entry-Level Network Engineer (CCNA/CCNP + Firewall)
description: Welcome to Network Consultancy Services (NCS), an esteemed Information Technology System Integrator established in 2009. With a focus on excellence and innovation, we specialize in providing comprehensive IT solutions to corporates and small and medium businesses. Our expertise spans across IT product sales, service, maintenance, and designing custom solutions tailored to meet your unique requirements. As a trusted partner, we offer a wide range of services, including IT engineer outsourcing and Annual Maintenance Services, ensuring the reliability and efficiency of your IT infrastructure. We are proud to deliver the latest technology solutions, including Artificial Intelligence, Internet of Things, Robots, Virtual Reality and more. With strategic partnerships and industry-leading certifications, we offer unparalleled quality and value. Our track record of success includes executing prestigious projects and establishing full IT backbone support contracts with major companies. With a team of multi-vendor certified experts, we bring a wealth of knowledge and experience to every project. From IT consultancy to taking on large-scale initiatives, we ensure seamless integration and optimal performance. Whether you are a corporate giant or a small business, our cutting-edge innovation solutions empower you to stay ahead in today's dynamic technological landscape. Choose NCS as your trusted IT partner, and experience the power of integrated technology solutions that drive success. Contact us today to embark on a transformative journey of innovation and growth with Network Consultancy Services.The RoleYou Will Be Responsible ForLeading a team of engineers to design, maintain and support the network infrastructure.Maximising system performance through proactive monitoring and ensuring reliability and availability.Recommending infrastructure solutions to meet business requirement in compliance with IT policy & procedure.Troubleshooting network issues and outages.Configuration, installation, maintenance and lifecycles planning of various network devices and services (routers, switches, firewalls, load balancers, VPN etc. .).Ideal ProfileYou possess a Degree/Diploma in Computer Science, Engineering or related field.You have at least 1 year experience, ideally within a Network Engineer / Systems Engineer role.You possess strong analytical skills and are comfortable dealing with numerical dataYou are adaptable and thrive in changing environmentsYou are highly goal driven and work well in fast paced environmentsYou are willing to undertake 0-30% travel.What's on Offer?Work alongside & learn from best in class talentLeadership RoleExcellent career development opportunities
requirements: 
career_level: Entry Level
```








