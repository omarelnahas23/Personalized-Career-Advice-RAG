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
<br><br>


# **Improving RAG Chain with Hybrid Search + Cohere Rerank**
**BM25** is a sophisticated ranking function used in information retrieval. Acting like a highly efficient librarian, it excels in navigating through extensive collections of documents. Its effectiveness lies in term Frequency: Evaluating how often search terms appear in each document. Document Length Normalization: Ensuring a fair chance for both short and long documents in search results. Bias-Free Information Retrieval: Ideal for large data sets where unbiased results are critical.<br>


**BM25 Retriever** - Sparse retriever <br>

**Embeddings** - Dense retrievers Milvus <br>

`Hybrid search = Sparse + Dense retriever`<br>

```python

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document


# Initialize the BM25 retriever
bm25_retriever = BM25Retriever.from_documents(doc_splits)
bm25_retriever.k = 1  

# Initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, Milvus_retriever], weights=[0.2, 0.8] # Retrieve top 2 results
)

# Example customer query
question = "Senior Machine Learning Engineer"


# Retrieve relevant documents/products
docs = ensemble_retriever.get_relevant_documents(question)

# Extract and print only the page content from each document
pretty_print_docs(docs)



```

<br><br>
output
```console
Document 1:

job_title: Senior Machine Learning Engineer
description: Siemens Digital Industries (DI) is an innovation leader in automation and digitalization. Closely, collaborating with partners and customers, we care about the digital transformation in the process and discrete industries. With our Digital Enterprise portfolio, we provide and encourage companies of all sizes with an end-to-end set of products, solutions and services to integrate and digitalize the entire value chain. Meaningful optimization for the specific needs of each industry, our outstanding portfolio supports customers to achieve greater efficiency and flexibility. We are constantly adding innovations to its portfolio to integrate groundbreaking future technologies. We have our global headquarters in Nuremberg, Germany, and have around 75,000 employees internationally.Are you passionate about advancing machine learning and software engineering? Join our Calibre SONR team in Cairo, Egypt, as a Senior Software Engineer / Technical Lead.Calibre SONR seamlessly integrates machine-learning models with the core Calibre architecture to enhance the productivity and precision of fab defect detection and diagnostics. This role offers a truly global scope and presents the opportunity to drive continuous improvement in one of our most critical services.ResponsibilitiesDesign, develop, and implement software programming for both internal and external products, exceeding customer expectations with a focus on quality and on-time delivery.Ensure the overall functional quality of released products across all platforms and mechanisms.Lead major projects within the product area, providing technical guidance and promoting innovation.Consult with customers on future upgrades and products, influencing technical direction.Provide high-level technical expertise, including in-depth software systems programming and analysis.Mentor junior engineers, demonstrating independence and technical expertise.QualificationsBachelor’s or Master’s degree in Computer Science, Computer Engineering, or a related field.+8 years of experience in software development, with a specialization in data analysis and machine learning.Proficiency in programming languages such as Python, Java, or C++.Experience with machine learning frameworks (e.g., TensorFlow, PyTorch).Strong understanding of data structures, algorithms, and software design principles.Excellent problem-solving skills and attention to detail.Highly developed communication skills, with the ability to present ideas and share knowledge effectively.Why us?Working at Siemens Software means flexibility - Choosing between working at home and the office at other times is the norm here. We offer great benefits and rewards, as you'd expect from a world leader in industrial software.We are an equal opportunity employer and value diversity at our company. We do not discriminate on the basis of race, religion, color, national origin, sex, gender, gender expression, sexual orientation, age, marital status, veteran status, or disability status.At Siemens, we are always challenging ourselves to build a better future. We need the most innovative and diverse Digital Minds to develop tomorrow‘s reality.Siemens Industry Software is an equal opportunities employer and does not discriminate unlawfully on the grounds of age, disability, gender assignment, marriage, and civil partnership, pregnancy and maternity, race, religion or belief, sex, sexual orientation, or trade union membership.If you want to make a difference – make it with us!
requirements: 
career_level: Not specified
----------------------------------------------------------------------------------------------------
Document 2:

job_title: Senior AI Researcher
description: <p>- Conduct state-of-the-art research in AI & Deep/Machine Learning applications in the field of Educational Technology<br>- Implement and prototype AI research ideas on Web & Mobile applications<br>- Innovate solutions in the domain of Educational Technology<br>- Publish research papers when requested<br>- [Senior] Mentors peers</p>
requirements: <p>- BSc in one of these fields: Computer Science, Computer Engineering, Electronics and Communications, Mechatronics, Biomedical or Bioinformatics<br>- MSc/PhD or enrollment in a&nbsp;post-graduate program (MSc/PhD) with focus on Deep Learning/Machine Learning is a plus<br>- 3+ years of relevant experience<br>- Strong Machine Learning background<br>- Deep Learning background<br>- Web &amp; Mobile App development background is a plus<br>- Experience in Machine Learning Cloud Deployment is a plus<br>&nbsp;</p>
career_level: Experienced (Non-Manager)
```


# **Using Reranking with [Cohere Reranker](https://cohere.com/rerank)**
```python
cohere_api_key = "<Your_API_Key>"
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere



compressor = CohereRerank(cohere_api_key=cohere_api_key)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)

compressed_docs = compression_retriever.invoke(
    question
)
pretty_print_docs(compressed_docs)
```
<br><br>
output
```console
Document 1:

job_title: Senior Machine Learning Engineer
description: Siemens Digital Industries (DI) is an innovation leader in automation and digitalization. Closely, collaborating with partners and customers, we care about the digital transformation in the process and discrete industries. With our Digital Enterprise portfolio, we provide and encourage companies of all sizes with an end-to-end set of products, solutions and services to integrate and digitalize the entire value chain. Meaningful optimization for the specific needs of each industry, our outstanding portfolio supports customers to achieve greater efficiency and flexibility. We are constantly adding innovations to its portfolio to integrate groundbreaking future technologies. We have our global headquarters in Nuremberg, Germany, and have around 75,000 employees internationally.Are you passionate about advancing machine learning and software engineering? Join our Calibre SONR team in Cairo, Egypt, as a Senior Software Engineer / Technical Lead.Calibre SONR seamlessly integrates machine-learning models with the core Calibre architecture to enhance the productivity and precision of fab defect detection and diagnostics. This role offers a truly global scope and presents the opportunity to drive continuous improvement in one of our most critical services.ResponsibilitiesDesign, develop, and implement software programming for both internal and external products, exceeding customer expectations with a focus on quality and on-time delivery.Ensure the overall functional quality of released products across all platforms and mechanisms.Lead major projects within the product area, providing technical guidance and promoting innovation.Consult with customers on future upgrades and products, influencing technical direction.Provide high-level technical expertise, including in-depth software systems programming and analysis.Mentor junior engineers, demonstrating independence and technical expertise.QualificationsBachelor’s or Master’s degree in Computer Science, Computer Engineering, or a related field.+8 years of experience in software development, with a specialization in data analysis and machine learning.Proficiency in programming languages such as Python, Java, or C++.Experience with machine learning frameworks (e.g., TensorFlow, PyTorch).Strong understanding of data structures, algorithms, and software design principles.Excellent problem-solving skills and attention to detail.Highly developed communication skills, with the ability to present ideas and share knowledge effectively.Why us?Working at Siemens Software means flexibility - Choosing between working at home and the office at other times is the norm here. We offer great benefits and rewards, as you'd expect from a world leader in industrial software.We are an equal opportunity employer and value diversity at our company. We do not discriminate on the basis of race, religion, color, national origin, sex, gender, gender expression, sexual orientation, age, marital status, veteran status, or disability status.At Siemens, we are always challenging ourselves to build a better future. We need the most innovative and diverse Digital Minds to develop tomorrow‘s reality.Siemens Industry Software is an equal opportunities employer and does not discriminate unlawfully on the grounds of age, disability, gender assignment, marriage, and civil partnership, pregnancy and maternity, race, religion or belief, sex, sexual orientation, or trade union membership.If you want to make a difference – make it with us!
requirements: 
career_level: Not specified
----------------------------------------------------------------------------------------------------
Document 2:

job_title: Senior AI Researcher
description: <p>- Conduct state-of-the-art research in AI & Deep/Machine Learning applications in the field of Educational Technology<br>- Implement and prototype AI research ideas on Web & Mobile applications<br>- Innovate solutions in the domain of Educational Technology<br>- Publish research papers when requested<br>- [Senior] Mentors peers</p>
requirements: <p>- BSc in one of these fields: Computer Science, Computer Engineering, Electronics and Communications, Mechatronics, Biomedical or Bioinformatics<br>- MSc/PhD or enrollment in a&nbsp;post-graduate program (MSc/PhD) with focus on Deep Learning/Machine Learning is a plus<br>- 3+ years of relevant experience<br>- Strong Machine Learning background<br>- Deep Learning background<br>- Web &amp; Mobile App development background is a plus<br>- Experience in Machine Learning Cloud Deployment is a plus<br>&nbsp;</p>
career_level: Experienced (Non-Manager)
```
<br><br>
# **Good To Keep in mind of Summarization**
Retrieval results may contain redundant or unnecessary information, potentially preventing LLMs
from generating accurate responses. Additionally, long prompts can slow down the inference process.
Therefore, efficient methods to summarize retrieved documents are crucial in the RAG pipeline.
Summarization tasks can be extractive or abstractive. Extractive methods segment text into sentences, then score and rank them based on importance. Abstractive compressors synthesize information from multiple documents to rephrase and generate a cohesive summary. These tasks can be
query-based or non-query-based. In this paper, as RAG retrieves information relevant to queries, we
focus exclusively on query-based methods.<br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/LLMLingua_logo.png)
<br><br>
I am using Latest LongLLMLingua for this LLMLingua2<br>
```python
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
)
docs_txt = []
for doc in compressed_docs:
    docs_txt.append(doc.page_content)
docs_txt = '\n\n'.join(docs_txt)
# 2000 Compression
compressed_prompt = llm_lingua.compress_prompt(
    docs_txt,
    rate=0.33,
    force_tokens=["!", ".", "?", "\n"],
    drop_consecutive=True,
)
# Reducing the size of the context of 33% 
```
<br>

**can be useful if we are retrieving big amount of job positing and we need to summerize the context for the model**
<br><br>
# **Evaluation of Pipelines Using [RAGAs](https://docs.ragas.io/en/latest/getstarted/index.html)**
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/Screenshot_1.png)

First of all, There are 2 models I am using for this evaluation in RAG Chain:<br>
1. **[Gemma 2 9B](https://blog.google/technology/developers/google-gemma-2/)**:<br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/Gemma2.png)
<br><br>
3.  **[LLama3 8B](https://ai.meta.com/blog/meta-llama-3/)**:<br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/LLama3.png)
<br><br>

## **Synthetic Test Set Generation**
We can leverage Ragas' [`Synthetic Test Data generation`](https://docs.ragas.io/en/stable/concepts/testset_generation.html) functionality to generate our own synthetic QC pairs - as well as a synthetic ground truth - quite easily!
<br><br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/SynTest.png)
<br><br>

Ragas is a novel approach to evaluation data generation that uses an evolutionary paradigm to create diverse samples of questions with varying difficulty levels. This method helps ensure comprehensive coverage of the performance of various components within a pipeline, resulting in a more robust evaluation process.<br>

The system works by taking a seed question and evolving it through a series of steps, each of which adds a new layer of complexity. These steps include:<br>

1. **Reasoning**: Rewriting the question to enhance the need for reasoning.<br>
2. **Conditioning**: Modifying the question to introduce a conditional element.<br>
3. **Multi-Context**: Rephrasing the question to require information from multiple related sections.<br>

The system also includes the ability to create conversational questions, which simulate a chat-based question-and-follow-up interaction.<br>

This process ensures that the generated test data is more representative of the questions that will be encountered in production, and therefore leads to a more accurate evaluation of the performance of LLMs.<br>
The Generation of the Synthetic Test Set is based on OpenAI GPT models so you need to have an api_key <br>

```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
import os
os.environ["OPENAI_API_KEY"] = "<OpenAi-Key>"

import pandas as pd
df = pd.read_csv("sampled_jobs.csv")
df = df.dropna(ignore_index=True) # Droping sample jobs positing that any nan value in any of its columns
df.to_csv("test.csv",index=False)
docs = CSVLoader("test.csv").load()

doc_splits_test = text_splitter.split_documents(docs)
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
import random
generator = TestsetGenerator.with_openai("gpt-4-turbo","gpt-4-turbo")

testset = generator.generate_with_langchain_docs(random.choices(doc_splits_test,k=20), test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
```
<br><br>
Note here the generator_llm and critic_llm is GPT-4-turbo model<br>
1. **Generator_LLM**:

**Purpose**: This LLM is responsible for generating new questions based on the provided documents. It takes as input the "seed question" and uses its vast knowledge and language understanding capabilities to create variations and complexities in the question. <br>
**Methods**: The generator_LLM might employ techniques like:<br>
    1. **Paraphrasing**: Rewording the original question to create different phrasing while retaining the core meaning.<br>
    2. **Adding Context**: Incorporating additional information or constraints into the question.<br>
    3. **Introducing Reasoning**: Making the question requires logical deductions or inferences to answer.<br>

2. **Critic_LLM**:

**Purpose**: This LLM acts as an evaluator, judging the quality of the generated questions. Its goal is to ensure that the generated questions are:<br>
        1. **Answerable**: The question can be answered based on the provided documents.<br>
        2. **Relevant**: The question remains aligned with the topic of the documents.<br>
        3. **Challenging**: The question requires some level of reasoning or understanding.<br>

**Methods**: The critic_LLM might use techniques like:<br>
        1. **Fact Checking**: Assessing if the generated question aligns with the information within the documents.<br>
        2. **Logical Analysis**: Evaluating the reasoning and coherence required to answer the question.<br>
        3. **Language Quality**: Checking for grammatical correctness and clarity in the generated question.<br>
<br><br>
**The Process**:<br>

The generator_LLM and critic_LLM work in an iterative loop:<br>

1. The generator_LLM proposes a new question.<br>
2. The critic_LLM evaluates the question for quality.<br>
3. If the question passes the critic_LLM's evaluation, it is added to the test set.<br>
4. If the question fails, the generator_LLM is prompted to modify or generate a new question.<br>
5. This back-and-forth process continues until the desired number of high-quality, diverse questions are generated for the test set.<br>
<br><br>
## **Evaluation Metrics**


1. [**Faithfulness**](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html): <br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/faithfullness.png) <br>
2. [**Answer Relevancy**](https://docs.ragas.io/en/stable/concepts/metrics/answer_relevance.html): <br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/Answer%20Relevancy.png) <br>
3. [**Context Precision**](https://docs.ragas.io/en/stable/concepts/metrics/context_precision.html): <br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/Context%20Precision.png) <br>
4. [**Context Recall**](https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html): <br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/Context%20Recall.png) <br>
5. [**Answer Correctness**](https://docs.ragas.io/en/stable/concepts/metrics/answer_correctness.html): <br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/Answer%20Correctness.png) <br>
<br><br>
can be imported using regas 0.1.0 like this:<br>
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_precision,
)

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
]
```
<br><br>
## **Evaluation Results**
There were 4 RAGs Chains Tested:<br>
1. **Gemma2 + HyDE** <br>
2. **Gemma2 + Hybrid Search + Cohere Reranker** <br>
3. **LLama3 + HyDE** <br>
4. **LLama3 + Hybrid Search + Cohere Reranker** <br>
<br><br>
Results are as follows:<br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/Hybrid%20Results.png)
<br><br>
![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/HyDE%20results.png)
<br><br>
**LLama3 + HyDE** Showed best performance in Faithfulness, Context Recall, Context Precision <br>
**LLama3 Hybrid ReRank** Showed best performance Answer Relevancy<br>
**Gemma2 + HyDE** Showed best performance same with **LLama3 + HyDE** in  Context Precision<br>
**Gemma2 + HyDE** Showed best performance in  Answer Correctness<br>
