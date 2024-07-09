![](https://github.com/omarelnahas23/Personalized-Career-Advice-RAG/blob/main/assets/Screenshot_8.png))]
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
```markdown

pip install -r requirements.txt
```
install Ollama and pull models
```markdown

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
LLMs face a series of challenges, including
issues such as hallucination [10, 33], instruction following [7, 58],
and handling long contexts [2, 8]. Many of these challenges can be
traced back to the inherent limitations of LLMs, with three critical
boundaries deserving attention.
• **Knowledge boundary**. LLMs are constrained by their knowledge capacity. Due to finite model parameters, they cannot fully
internalize the vast body of world knowledge. Moreover, the internal knowledge of LLMs is static and difficult to be updated with
the dynamically evolving world. Furthermore, LLMs are predominantly trained on publicly available, high-frequency data, which
may result in inaccuracies when dealing with domain-specific or
long-tail knowledge.
• **Memory boundary**. LLMs also grapple with severe limitations
in memory, primarily due to restrictions on context length. While
advances have been continually made in expanding the maximum
context length, it still falls short of achieving the goal of lifelong
engagement with human users. Additionally, both the training and
deployment of LLMs with extended context can be prohibitively
computationally and storage-intensive, making it impractical to
significantly expand their memory.
• **Capability boundary**. LLMs’ capabilities are constrained
in terms of action and autonomy. Firstly, they are limited to the
’language space’ and cannot meaningfully interact with the physical
world. Secondly, these models heavily rely on human guidance,
requiring clear user instructions and appropriate demonstration
examples to perform specific tasks effectively.
