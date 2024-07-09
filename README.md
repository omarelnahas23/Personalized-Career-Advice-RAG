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
