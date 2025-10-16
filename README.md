# AI Hero RAG Agent with Streamlit & Pydantic AI

A production-ready Retrieval-Augmented Generation (RAG) system that transforms any GitHub repository into an intelligent, conversational AI assistant. Built with Pydantic AI for robust agent orchestration and Streamlit for an intuitive web interface.

## 🎯 Project Overview

This project implements a complete RAG pipeline that:
- Downloads and processes documentation from GitHub repositories
- Chunks large documents intelligently for optimal retrieval
- Indexes content using both lexical (text) and semantic (vector) search
- Creates an AI agent with tool-calling capabilities
- Provides systematic evaluation and logging
- Deploys as a web application with streaming responses

**Built following the AI Agents Crash Course** - a comprehensive 7-day curriculum covering data ingestion, chunking strategies, search implementation, agent development, evaluation systems, and deployment.

## 🚀 Who Benefits From This Project?

### For Developers & Technical Teams
- **Developer Tool Creators**: Build intelligent documentation assistants for your libraries, frameworks, or APIs
- **Open Source Maintainers**: Help users navigate complex documentation and reduce support burden
- **DevOps Teams**: Create internal knowledge bases for infrastructure documentation and runbooks
- **Technical Writers**: Augment documentation with interactive Q&A capabilities

### For Organizations & Businesses
- **Customer Support Teams**: Automate first-line support with accurate, source-cited responses
- **Training Departments**: Create interactive learning assistants for employee onboarding
- **Compliance Teams**: Build searchable policy and procedure databases with audit trails
- **Research Groups**: Enable efficient exploration of large document repositories

### For Educators & Communities
- **Course Instructors**: Provide 24/7 teaching assistants for students
- **Online Learning Platforms**: Scale support without increasing headcount
- **Technical Communities**: Help members find answers in wikis and knowledge bases
- **Bootcamp Organizers**: Reduce repetitive FAQ questions with intelligent automation

### Key Value Propositions
- **Cost Reduction**: Automate responses to routine questions, freeing human experts for complex issues
- **24/7 Availability**: Never miss a question, regardless of timezone or business hours
- **Consistent Accuracy**: Responses always reference source material, reducing misinformation
- **Scalability**: Handle unlimited simultaneous queries without degradation
- **Knowledge Preservation**: Capture and structure organizational knowledge for long-term retention
- **Developer Experience**: Clean, modular codebase that's easy to customize and extend

## 📁 Repository Structure

```
Aihero-RAG-Streamlit-Pydantic/
├── app/                    # Production deployment code
│   ├── pyproject.toml      # uv project configuration & dependencies
│   ├── app.py              # Streamlit web interface with streaming
│   ├── ingest.py           # Data loading, chunking, and indexing
│   ├── search_tools.py     # Search tool implementation
│   ├── search_agent.py     # Pydantic AI agent configuration
│   ├── logs.py             # Conversation logging system
│   └── main.py             # CLI interface for testing
│
└── course/                 # Learning materials & experiments
    ├── pyproject.toml      # Development dependencies
    └── notebooks/          # Jupyter notebooks for each lesson
```

## 🛠️ Technology Stack

- **Agent Framework**: Pydantic AI (type-safe agent orchestration)
- **Search Engine**: minsearch (in-memory text & vector search)
- **Embeddings**: sentence-transformers (semantic similarity)
- **Web Framework**: Streamlit (rapid UI development)
- **LLM Provider**: OpenAI GPT-4o-mini (function calling & chat)
- **Package Manager**: uv (modern Python dependency management)

## 📚 Understanding uv and Virtual Environments

### What is uv?

`uv` is a modern Python package and project manager written in Rust. It's significantly faster than traditional tools like pip and provides a superior developer experience.

**Why use uv over pip?**
- **Speed**: 10-100x faster than pip for dependency resolution and installation
- **Reliability**: Better dependency resolver that prevents conflicts
- **Simplicity**: Combines functionality of pip, virtualenv, and pip-tools in one tool
- **Modern**: Built with current Python best practices in mind
- **Project Management**: Handles virtual environments automatically

### Virtual Environments Explained

A virtual environment is an isolated Python workspace that keeps your project's dependencies separate from your system Python and other projects.

**Why virtual environments matter:**
- **Dependency Isolation**: Each project has its own package versions without conflicts
- **Reproducibility**: Anyone can recreate your exact development environment
- **Safety**: Experiments won't break your system Python or other projects
- **Version Management**: Use different Python versions for different projects
- **Clean Uninstalls**: Delete the environment folder to remove everything

**Traditional approach (without uv):**
```bash
python -m venv myenv          # Create environment
source myenv/bin/activate     # Activate it (Linux/Mac)
myenv\Scripts\activate        # Activate it (Windows)
pip install package           # Install packages
deactivate                    # Exit environment
```

**With uv (simpler):**
```bash
uv init                       # Creates project + environment automatically
uv add package                # Adds package and updates lock file
uv run python script.py       # Runs in virtual environment automatically
```

### Understanding pyproject.toml

The `pyproject.toml` file is the modern standard for Python project configuration. It replaces older files like `setup.py`, `requirements.txt`, and `setup.cfg`.

**Structure breakdown:**

```toml
[project]
name = "aihero-rag"                    # Your project name
version = "0.1.0"                       # Semantic versioning
description = "RAG agent system"        # Brief description
readme = "README.md"                    # Documentation file
requires-python = ">=3.10"              # Minimum Python version

dependencies = [                        # Runtime dependencies
    "minsearch>=0.0.5",                # Search engine
    "openai>=1.108.2",                 # LLM API client
    "pydantic-ai==1.0.9",              # Agent framework (exact version)
    "python-frontmatter>=1.1.0",       # Markdown metadata parser
    "requests>=2.32.5",                # HTTP library
]

[project.optional-dependencies]
dev = [                                 # Development-only dependencies
    "jupyter>=1.0.0",                  # Interactive notebooks
    "ipykernel>=6.0.0",                # Jupyter kernel
]

[build-system]
requires = ["hatchling"]                # Build tool
build-backend = "hatchling.build"       # Build backend
```

**Dependency specification formats:**

- `package>=1.0.0` - Minimum version (allows 1.1.0, 2.0.0, etc.)
- `package==1.0.0` - Exact version (only 1.0.0)
- `package>=1.0.0,<2.0.0` - Version range
- `package[extra]` - Include optional features

**Why use pyproject.toml?**
- **Standards-Based**: PEP 518 standardized format
- **All-in-One**: Replaces multiple configuration files
- **Tool Support**: Works with modern tools (uv, Poetry, pip-tools)
- **Human-Readable**: TOML is easy to read and edit
- **Dependency Locking**: Works with lock files for exact reproducibility

### The uv Workflow

**1. Starting a new project:**
```bash
uv init my-project
cd my-project
```

This creates:
- `pyproject.toml` - Project configuration
- `.python-version` - Python version specification
- A virtual environment (hidden in `uv` cache)

**2. Adding dependencies:**
```bash
# Runtime dependency
uv add requests

# Development dependency
uv add --dev pytest

# Specific version
uv add "pydantic-ai==1.0.9"

# Multiple at once
uv add openai streamlit minsearch
```

Each `uv add` command:
- Installs the package in your virtual environment
- Updates `pyproject.toml` with the dependency
- Updates `uv.lock` with exact versions of all packages

**3. Running your code:**
```bash
# Run Python scripts
uv run python my_script.py

# Run installed CLI tools
uv run streamlit run app.py

# Execute Jupyter
uv run jupyter notebook
```

The `uv run` command automatically:
- Activates the virtual environment
- Ensures all dependencies are installed
- Runs your command
- Cleans up afterward

**4. Syncing dependencies:**
```bash
# Install all dependencies from pyproject.toml
uv sync

# Install including dev dependencies
uv sync --all-extras
```

**5. Exporting for compatibility:**
```bash
# Create requirements.txt for platforms that don't support uv
uv export --no-dev > requirements.txt

# Include dev dependencies
uv export > requirements-dev.txt
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Git (for cloning the repository)

### Step-by-Step Setup

**1. Install uv**

Choose your platform:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: using pip (if you have Python already)
pip install uv
```

**2. Clone the repository**

```bash
git clone https://github.com/denis911/Aihero-RAG-Streamlit-Pydantic.git
cd Aihero-RAG-Streamlit-Pydantic
```

**3. Set up the application environment**

```bash
# Navigate to app directory
cd app

# Initialize and sync dependencies
uv sync

# This automatically:
# - Creates a virtual environment
# - Installs all dependencies from pyproject.toml
# - Sets up the project structure
```

**4. Configure your OpenAI API key**

You have several options:

**Option A: Environment variable (recommended for development)**
```bash
# Linux/Mac
export OPENAI_API_KEY='your-api-key-here'

# Windows (Command Prompt)
set OPENAI_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"
```

**Option B: Using direnv (best for persistent configuration)**
```bash
# Install direnv (one-time setup)
# macOS: brew install direnv
# Linux: apt-get install direnv

# Create .envrc file
echo 'export OPENAI_API_KEY="your-api-key-here"' > .envrc
direnv allow

# Add to .gitignore (IMPORTANT!)
echo ".envrc" >> .gitignore
```

**Option C: Streamlit secrets (for deployment)**
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-api-key-here"
```

**Important Security Notes:**
- Never commit API keys to version control
- Always add `.envrc`, `.env`, and `.streamlit/secrets.toml` to `.gitignore`
- Use project-specific API keys in OpenAI dashboard to limit exposure
- Monitor your API usage regularly

## 🚀 Usage

### Running the Web Application

```bash
# Make sure you're in the app directory
cd app

# Launch Streamlit
uv run streamlit run app.py

# Your browser will open automatically to http://localhost:8501
```

**Features:**
- Interactive chat interface with streaming responses
- Automatic search tool invocation
- Source citations with GitHub links
- Conversation logging for evaluation
- Session state management

### Using the Command-Line Interface

For testing without the web UI:

```bash
cd app
uv run python main.py
```

This provides:
- Simple question-answer loop
- Console output for debugging
- Full logging functionality
- Type 'stop' to exit

### Exploring the Course Materials

```bash
# Navigate to course directory
cd course

# Install development dependencies
uv sync

# Start Jupyter
uv run jupyter notebook
```


## 🔍 How It Works

### 1. Data Ingestion (ingest.py)

The system downloads GitHub repositories as ZIP archives and processes markdown files:

```python
def read_repo_data(repo_owner, repo_name):
    # Downloads repo as ZIP (no git clone needed)
    # Extracts all .md and .mdx files
    # Parses frontmatter metadata
    # Returns structured documents
```

**Why ZIP instead of git clone?**
- Faster and lighter weight
- No git history needed
- Works in memory without disk writes
- Easier to deploy in cloud environments

### 2. Chunking Strategy (ingest.py)

Large documents are split using a sliding window approach:

```python
def sliding_window(seq, size=2000, step=1000):
    # Creates overlapping chunks
    # Preserves context at boundaries
    # Prevents information loss
```

**Why overlap chunks?**
- Important information often spans boundaries
- Improves retrieval quality
- Maintains context for better understanding
- Standard practice in RAG systems

### 3. Search Implementation (search_tools.py)

The search tool uses minsearch for in-memory indexing:

```python
class SearchTool:
    def search(self, query: str) -> List[Any]:
        # Text-based search for keyword matching
        # Returns top 5 most relevant documents
        # Can be extended to vector/hybrid search
```

**Search options:**
- **Text Search**: Fast keyword matching, good for exact terms
- **Vector Search**: Semantic similarity, handles paraphrasing
- **Hybrid Search**: Combines both for best results

### 4. Agent Configuration (search_agent.py)

Pydantic AI orchestrates the agent with function calling:

```python
def init_agent(index, repo_owner, repo_name):
    # Creates agent with system prompt
    # Registers search tool
    # Configures GPT-4o-mini model
    # Enables function calling
```

**Agent capabilities:**
- Analyzes user questions
- Decides when to search
- Invokes search tool with appropriate queries
- Synthesizes answers from results
- Cites sources with GitHub links

### 5. Logging & Evaluation (logs.py)

Every interaction is logged for systematic evaluation:

```python
def log_interaction_to_file(agent, messages, source='user'):
    # Captures full conversation history
    # Records tool calls and responses
    # Saves with timestamps
    # Enables LLM-as-judge evaluation
```

**What gets logged:**
- System prompt and model configuration
- User questions
- Tool invocations and results
- Final agent responses
- Timestamps and metadata

### 6. Streamlit Interface (app.py)

The web UI provides streaming responses:

```python
async with agent.run_stream(user_prompt=prompt) as result:
    async for chunk in result.stream_output():
        # Streams tokens as they're generated
        # Updates UI in real-time
        # Better user experience
```

**UI features:**
- Chat history persistence (session state)
- Streaming response display
- Loading indicators
- Mobile-responsive design

## 📊 Customization Guide

### Using Your Own Repository

Edit `app.py` or `main.py`:

```python
# Change these constants
REPO_OWNER = "your-username"
REPO_NAME = "your-repo"

# Optional: Add filtering
def filter(doc):
    return 'specific-folder' in doc['filename']

index = ingest.index_data(REPO_OWNER, REPO_NAME, filter=filter)
```

### Adjusting Chunking Parameters

In `ingest.py`:

```python
# Smaller chunks for more precise retrieval
chunks = chunk_documents(docs, size=1000, step=500)

# Larger chunks for more context
chunks = chunk_documents(docs, size=3000, step=1500)

# No overlap (not recommended)
chunks = chunk_documents(docs, size=2000, step=2000)
```

**Guidelines:**
- Start with defaults (2000/1000)
- Decrease size for Q&A with short answers
- Increase size for narrative content
- Always maintain some overlap

### Customizing System Prompt

In `search_agent.py`:

```python
SYSTEM_PROMPT_TEMPLATE = """
You are a [YOUR ROLE] assistant.

[YOUR INSTRUCTIONS]

Always cite sources using: [TEXT](GITHUB_LINK)
"""
```

**Prompt engineering tips:**
- Be specific about desired behavior
- Include examples of good responses
- Specify citation format clearly
- Test with evaluation dataset

### Changing the LLM Model

In `search_agent.py`:

```python
agent = Agent(
    model='gpt-4o',  # More capable, higher cost
    # model='gpt-3.5-turbo',  # Faster, cheaper
    # model='gpt-4o-mini',  # Default: balanced
)
```

**Model selection:**
- `gpt-4o-mini`: Best value, good for most use cases
- `gpt-4o`: Highest quality, more expensive
- `gpt-3.5-turbo`: Budget option, faster responses

### Implementing Vector Search

Add to `search_tools.py`:

```python
from sentence_transformers import SentenceTransformer

class SearchTool:
    def __init__(self, index, embeddings, embedding_model):
        self.index = index
        self.embeddings = embeddings
        self.embedding_model = embedding_model
    
    def vector_search(self, query: str) -> List[Any]:
        # Encode query
        q_vector = self.embedding_model.encode(query)
        # Search with vector similarity
        return self.vector_index.search(q_vector, num_results=5)
```

## 🧪 Evaluation & Testing

### Manual Testing

```bash
cd app
uv run python main.py
```

Ask various questions and inspect:
- Answer relevance and accuracy
- Source citations
- Tool usage patterns
- Response quality

### Automated Evaluation

The course materials include LLM-as-judge evaluation:

1. **Collect logs** from user interactions
2. **Generate test data** with AI
3. **Run evaluation agent** with checklist:
   - Instructions followed?
   - Answer relevant?
   - Citations included?
   - Search tool used?
   - Response complete?

4. **Analyze metrics** with pandas

See Day 5 materials for complete evaluation implementation.

## 🌐 Deployment

### Streamlit Community Cloud

**Prerequisites:**
- GitHub account
- Streamlit account ([sign up free](https://streamlit.io/cloud))
- Push your code to GitHub

**Steps:**

1. **Export dependencies:**
```bash
cd app
uv export --no-dev > requirements.txt
```

2. **Configure secrets:**

Create `.streamlit/secrets.toml` (local testing):
```toml
OPENAI_API_KEY = "your-key"
```

3. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file: `app/app.py`
   - Add secrets in dashboard
   - Deploy!

**Cost management:**
- Use project-specific OpenAI keys
- Set spending limits in OpenAI dashboard
- Monitor usage regularly
- Consider hibernating during low-traffic periods

### Alternative Deployment Options

**Docker:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN pip install uv
COPY . .
RUN uv sync
CMD ["uv", "run", "streamlit", "run", "app.py"]
```

**AWS/GCP/Azure:**
- Deploy as container service
- Use managed secrets for API keys
- Set up load balancing for scale
- Enable logging and monitoring

## 📈 Performance Optimization

### Caching Strategies

Streamlit provides built-in caching:

```python
@st.cache_resource
def init_agent():
    # Only runs once
    # Cached across all users
    # Persists between reruns
```

**What to cache:**
- Agent initialization
- Index creation
- Embedding model loading
- Static configuration

**What NOT to cache:**
- User inputs
- Conversation history
- Session-specific data

### Search Optimization

**For better retrieval:**
- Tune chunk size based on content
- Implement hybrid search
- Use query expansion techniques
- Add metadata filtering

**For better performance:**
- Precompute embeddings offline
- Use approximate nearest neighbors
- Implement result caching
- Batch process documents

### Cost Optimization

**Reduce API costs:**
- Use `gpt-4o-mini` instead of `gpt-4o`
- Implement response caching
- Optimize prompt length
- Set max_tokens limits

**Monitor usage:**
```python
import logging

logging.info(f"Prompt tokens: {response.usage.prompt_tokens}")
logging.info(f"Completion tokens: {response.usage.completion_tokens}")
```

## 🐛 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
# Ensure dependencies are installed
cd app
uv sync

# Check virtual environment
uv run python -c "import sys; print(sys.prefix)"
```

**OpenAI API errors:**
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API key
uv run python -c "import openai; print(openai.OpenAI().models.list())"
```

**Streamlit not finding modules:**
```bash
# Export requirements.txt
uv export --no-dev > requirements.txt

# Verify requirements
cat requirements.txt
```

**Rate limiting errors:**
- Implement exponential backoff
- Add retry logic
- Use rate limiting libraries
- Monitor API usage dashboard

### Debug Mode

Enable verbose logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 📚 Learning Resources

- **Pydantic AI Documentation**: [https://ai.pydantic.dev](https://ai.pydantic.dev)
- **Streamlit Documentation**: [https://docs.streamlit.io](https://docs.streamlit.io)
- **uv Documentation**: [https://docs.astral.sh/uv](https://docs.astral.sh/uv)
- **OpenAI API Reference**: [https://platform.openai.com/docs](https://platform.openai.com/docs)
- **RAG Best Practices**: [https://www.pinecone.io/learn/retrieval-augmented-generation](https://www.pinecone.io/learn/retrieval-augmented-generation)

## 🙏 Acknowledgments

This project is based on the **AI Agents Crash Course** by Alexey Grigorev. The course provides comprehensive instruction on building production-ready RAG systems from scratch.

- Course website: [https://alexeygrigorev.com/aihero](https://alexeygrigorev.com/aihero)
- Community: [DataTalks.Club Slack](https://datatalks.club) (#course-ai-bootcamp channel)

