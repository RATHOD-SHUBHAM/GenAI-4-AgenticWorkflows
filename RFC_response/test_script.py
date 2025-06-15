import nest_asyncio
import pickle
from pathlib import Path
import logging
import json
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict
from pydantic import BaseModel

# Load the environment
load_dotenv()

os.environ['llama_cloud'] = os.getenv('llama_cloud')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

from llama_cloud_services import LlamaParse
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole, LLM
from llama_index.utils.workflow import draw_all_possible_flows

# Configure logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# Constants and Prompts
# this is the research agent's system prompt, tasked with answering a specific question
AGENT_SYSTEM_PROMPT = """\
You are a research agent tasked with filling out a specific form key/question with the appropriate value, given a bank of context.
You are given a specific form key/question. Think step-by-step and use the existing set of tools to help answer the question.

You MUST always use at least one tool to answer each question. Only after you've determined that existing tools do not \
answer the question should you try to reason from first principles and prior knowledge to answer the question.

You MUST try to answer the question instead of only saying 'I dont know'.

"""

# This is the prompt tasked with extracting information from an RFP file.
EXTRACT_KEYS_PROMPT = """\
You are provided an entire RFP document, or a large subsection from it.

We wish to generate a response to the RFP in a way that adheres to the instructions within the RFP, \
including the specific sections that an RFP response should contain, and the content that would need to go \
into each section.

Your task is to extract out a list of "questions", where each question corresponds to a specific section that is required in the RFP response.
Put another way, after we extract out the questions we will go through each question and answer each one \
with our downstream research assistant, and the combined
question:answer pairs will constitute the full RFP response.

You must TRY to extract out questions that can be answered by the provided knowledge base. We provide the list of file metadata below.

Additional requirements:
- Try to make the questions SPECIFIC given your knowledge of the RFP and the knowledge base. Instead of asking a question like \
"How do we ensure security" ask a question that actually addresses a security requirement in the RFP and can be addressed by the knowledge base.
- Make sure the questions are comprehensive and addresses all the RFP requirements.
- Make sure each question is descriptive - this gives our downstream assistant context to fill out the value for that question
- Extract out all the questions as a list of strings.


Knowledge Base Files:
{file_metadata}

RFP Full Template:
{rfp_text}

"""

# this is the prompt that generates the final RFP response given the original template text and question-answer pairs.
GENERATE_OUTPUT_PROMPT = """\
You are an expert analyst.
Your task is to generate an RFP response according to the given RFP and question/answer pairs.

You are given the following RFP and qa pairs:

<rfp_document>
{output_template}
</rfp_document>

<question_answer_pairs>
{answers}
</question_answer_pairs>

Not every question has an appropriate answer. This is because the agent tasked with answering the question did not have the right context to answer it.
If this is the case, you MUST come up with an answer that is reasonable. You CANNOT say that you are unsure in any area of the RFP response.
Dont give any points or steps, keep the answer as paragraphs.

Never say you were not able to get the answer, come up with your own answer in such cases
Please generate the output according to the template and the answers, in markdown format.
Directly output the generated markdown content, do not add any additional text, such as "```markdown" or "Here is the output:".
Follow the original format of the template as closely as possible, and fill in the answers into the appropriate sections.
"""

# Event Classes
class OutputQuestions(BaseModel):
    questions: List[str]

class OutputTemplateEvent(Event):
    docs: List[Document]

class QuestionsExtractedEvent(Event):
    questions: List[str]

class HandleQuestionEvent(Event):
    question: str

class QuestionAnsweredEvent(Event):
    question: str
    answer: str

class CollectedAnswersEvent(Event):
    combined_answers: str

class LogEvent(Event):
    msg: str
    delta: bool = False

# Main Workflow Class
class RFPWorkflow(Workflow):
    def __init__(
        self,
        tools,
        parser: LlamaParse,
        llm: LLM | None = None,
        similarity_top_k: int = 20,
        output_dir: str = "data_out_rfp",
        agent_system_prompt: str = AGENT_SYSTEM_PROMPT,
        generate_output_prompt: str = GENERATE_OUTPUT_PROMPT,
        extract_keys_prompt: str = EXTRACT_KEYS_PROMPT,
        **kwargs,
    ) -> None:
        # ... existing initialization code ...
        """Init params."""
        super().__init__(**kwargs)
        self.tools = tools

        self.parser = parser

        self.llm = llm or OpenAI(model="gpt-4o-mini")
        self.similarity_top_k = similarity_top_k

        self.output_dir = output_dir

        self.agent_system_prompt = agent_system_prompt
        self.extract_keys_prompt = extract_keys_prompt

        # if not exists, create
        out_path = Path(self.output_dir) / "workflow_output"
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
            os.chmod(str(out_path), 0o0777)

        self.generate_output_prompt = PromptTemplate(generate_output_prompt)

    @step
    async def parse_output_template(self, ctx: Context, ev: StartEvent) -> OutputTemplateEvent:
        # ... existing code ...
        # load output template file
        out_template_path = Path(
            f"{self.output_dir}/workflow_output/output_template.jsonl"
        )
        if out_template_path.exists():
            with open(out_template_path, "r") as f:
                docs = [Document.model_validate_json(line) for line in f]
        else:
            docs = await self.parser.aload_data(ev.rfp_template_path)
            # save output template to file
            with open(out_template_path, "w") as f:
                for doc in docs:
                    print("Doc is here: ", doc)
                    f.write(doc.model_dump_json())
                    f.write("\n")

        await ctx.set("output_template", docs)
        return OutputTemplateEvent(docs=docs)

    @step
    async def extract_questions(self, ctx: Context, ev: OutputTemplateEvent) -> HandleQuestionEvent:
        # ... existing code ...
        docs = ev.docs

        # save all_questions to file
        out_keys_path = Path(f"{self.output_dir}/workflow_output/all_keys.txt")
        if out_keys_path.exists():
            with open(out_keys_path, "r") as f:
                output_qs = [q.strip() for q in f.readlines()]
        else:
            # try stuffing all text into the prompt
            all_text = "\n\n".join([d.get_content(metadata_mode="all") for d in docs])
            prompt = PromptTemplate(template=self.extract_keys_prompt)

            file_metadata = "\n\n".join(
                [
                    f"Name:{t.metadata.name}\nDescription:{t.metadata.description}"
                    for t in self.tools
                ]
            )
            try:
                if self._verbose:
                    ctx.write_event_to_stream(
                        LogEvent(msg=">> Extracting questions from LLM")
                    )

                output_qs = self.llm.structured_predict(
                    OutputQuestions,
                    prompt,
                    file_metadata=file_metadata,
                    rfp_text=all_text,
                ).questions

                if self._verbose:
                    qs_text = "\n".join([f"* {q}" for q in output_qs])
                    ctx.write_event_to_stream(LogEvent(msg=f">> Questions:\n{qs_text}"))

            except Exception as e:
                _logger.error(f"Error extracting questions from page: {all_text}")
                _logger.error(e)

            with open(out_keys_path, "w") as f:
                f.write("\n".join(output_qs))

        await ctx.set("num_to_collect", len(output_qs))

        for question in output_qs:
            ctx.send_event(HandleQuestionEvent(question=question))

        return None

    @step
    async def handle_question(self, ctx: Context, ev: HandleQuestionEvent) -> QuestionAnsweredEvent:
        # ... existing code ...
        question = ev.question

        # initialize a Function Calling "research" agent where given a task, it can pull responses from relevant tools and synthesize over it
        research_agent = FunctionCallingAgentWorker.from_tools(
            self.tools, llm=self.llm, verbose=False, system_prompt=self.agent_system_prompt
        ).as_agent()

        # ensure the agent's memory is cleared
        response = await research_agent.aquery(question)

        if self._verbose:
            # instead of printing the message directly, write the event to stream!
            msg = f">> Asked question: {question}\n>> Got response: {str(response)}"
            ctx.write_event_to_stream(LogEvent(msg=msg))

        return QuestionAnsweredEvent(question=question, answer=str(response))

    @step
    async def combine_answers(self, ctx: Context, ev: QuestionAnsweredEvent) -> CollectedAnswersEvent:
        # ... existing code ...
        num_to_collect = await ctx.get("num_to_collect")
        results = ctx.collect_events(ev, [QuestionAnsweredEvent] * num_to_collect)
        if results is None:
            return None

        combined_answers = "\n".join([result.model_dump_json() for result in results])
        # save combined_answers to file
        with open(
            f"{self.output_dir}/workflow_output/combined_answers.jsonl", "w"
        ) as f:
            f.write(combined_answers)

        return CollectedAnswersEvent(combined_answers=combined_answers)

    @step
    async def generate_output(self, ctx: Context, ev: CollectedAnswersEvent) -> StopEvent:
        # ... existing code ...
        output_template = await ctx.get("output_template")
        output_template = "\n".join(
            [doc.get_content("none") for doc in output_template]
        )

        if self._verbose:
            ctx.write_event_to_stream(LogEvent(msg=">> GENERATING FINAL OUTPUT"))

        resp = await self.llm.astream(
            self.generate_output_prompt,
            output_template=output_template,
            answers=ev.combined_answers,
        )

        final_output = ""
        async for r in resp:
            ctx.write_event_to_stream(LogEvent(msg=r, delta=True))
            final_output += r

        # save final_output to file
        with open(f"{self.output_dir}/workflow_output/final_output.md", "w") as f:
            f.write(final_output)

        return StopEvent(result=final_output)

def setup_environment():
    """Setup initial environment and download required files"""
    nest_asyncio.apply()
    
    # Create directories with proper permissions
    data_dir = Path("data")
    data_out_dir = Path("data_out_rfp")
    data_dir.mkdir(exist_ok=True, mode=0o777)
    data_out_dir.mkdir(exist_ok=True, mode=0o777)
    
    # Create storage directory for ChromaDB
    storage_dir = Path("storage_rfp_chroma")
    storage_dir.mkdir(exist_ok=True, mode=0o777)
    
    files = ["azure_gov.pdf", "azure_wiki.pdf", "msft_10k_2024.pdf", "msft_ddr.pdf"]
    
    # Verify files exist
    for file in files:
        file_path = data_dir / file
        if not file_path.exists():
            raise FileNotFoundError(f"Required file {file} not found in {data_dir}")
    
    return str(data_dir), str(data_out_dir), files

def parse_and_generate_summaries(data_dir: str, data_out_dir: str, files: List[str], parser: LlamaParse) -> Dict:
    """Parse files and generate summaries"""
    file_dicts = {}
    summary_llm = OpenAI(model="gpt-4o-mini")  # Using gpt-4o-mini instead of gpt-4-turbo-preview
    
    # Check if cached file exists
    cache_path = Path(data_out_dir) / "tmp_file_dicts.pkl"
    if cache_path.exists():
        try:
            return pickle.load(open(cache_path, "rb"))
        except Exception as e:
            _logger.warning(f"Failed to load cache, regenerating: {e}")

    # First, parse all documents
    _logger.info("Parsing all documents...")
    for f in files:
        file_base = Path(f).stem
        full_file_path = str(Path(data_dir) / f)
        
        # Parse documents
        file_docs = parser.load_data(full_file_path)
        
        # Attach metadata
        for idx, d in enumerate(file_docs):
            d.metadata["file_path"] = f
            d.metadata["page_num"] = idx + 1
            
        file_dicts[f] = {"file_path": full_file_path, "docs": file_docs}
    
    # Then, generate summaries for all files
    _logger.info("Generating summaries...")
    all_docs = []
    for f in files:
        all_docs.extend(file_dicts[f]["docs"])
    
    # Create a single index for all documents
    combined_index = SummaryIndex(all_docs)
    
    # Generate summaries for each file using metadata filters
    for f in files:
        _logger.info(f"Generating summary for {f}")
        # Create a filtered query engine for this file
        query_engine = combined_index.as_query_engine(
            llm=summary_llm,
            similarity_top_k=3,
            node_postprocessors=[
                lambda nodes: [n for n in nodes if n.metadata["file_path"] == f]
            ]
        )
        
        try:
            response = query_engine.query(
                "Generate a short 1-2 line summary of this file to help inform an agent on what this file is about."
            )
            _logger.info(f"Generated summary for {f}: {str(response)}")
            file_dicts[f]["summary"] = str(response)
        except Exception as e:
            _logger.error(f"Failed to generate summary for {f}: {e}")
            file_dicts[f]["summary"] = f"Summary generation failed for {f}"
    
    # Cache results
    pickle.dump(file_dicts, open(cache_path, "wb"))
    
    return file_dicts

def initialize_parser():
    """Initialize LlamaParse"""
    return LlamaParse(
        api_key = os.environ['llama_cloud'],
        result_type="markdown",
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="openai-gpt-4o-mini",
        vendor_multimodal_api_key=os.environ['OPENAI_API_KEY']
    )

def initialize_index(files: List[str], file_dicts: Dict, parser: LlamaParse) -> VectorStoreIndex:
    """Initialize the vector store index with the given files."""
    persist_dir = Path("storage_rfp_chroma")
    
    try:
        # Create or load vector store
        vector_store = ChromaVectorStore.from_params(
            collection_name="rfp_docs", 
            persist_dir=str(persist_dir)
        )
        
        # Load existing index if it exists
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Insert documents if directory doesn't exist or is empty
        if not persist_dir.exists() or not any(persist_dir.iterdir()):
            _logger.info("Inserting documents into new index...")
            all_nodes = [c for d in file_dicts.values() for c in d["docs"]]
            index.insert_nodes(all_nodes)
            
        return index
    except Exception as e:
        _logger.error(f"Failed to initialize index: {e}")
        raise

def create_file_descriptions(file_dicts: Dict) -> Dict[str, Dict[str, str]]:
    """Create descriptions for each file using generated summaries."""
    return {
        file: {"summary": file_info["summary"]} 
        for file, file_info in file_dicts.items()
    }

def generate_tool(index: VectorStoreIndex, file: str, file_description: Optional[str] = None):
    """Return a function that retrieves only within a given file."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="file_path", operator=FilterOperator.EQ, value=file),
        ]
    )

    def chunk_retriever_fn(query: str) -> str:
        retriever = index.as_retriever(similarity_top_k=5, filters=filters)
        nodes = retriever.retrieve(query)

        full_text = "\n\n========================\n\n".join(
            [n.get_content(metadata_mode="all") for n in nodes]
        )

        return full_text

    # define name as a function of the file
    fn_name = Path(file).stem + "_retrieve"

    tool_description = f"Retrieves a small set of relevant document chunks from {file}."
    if file_description is not None:
        tool_description += f"\n\nFile Description: {file_description}"

    tool = FunctionTool.from_defaults(
        fn=chunk_retriever_fn, name=fn_name, description=tool_description
    )

    return tool

async def main():
    try:
        # Setup environment
        data_dir, data_out_dir, files = setup_environment()
        
        # Initialize parser
        parser = initialize_parser()
        
        # Parse files and generate summaries
        file_dicts = parse_and_generate_summaries(data_dir, data_out_dir, files, parser)
        
        # Initialize index
        index = initialize_index(files, file_dicts, parser)
        
        # Create file descriptions using generated summaries
        file_descriptions = create_file_descriptions(file_dicts)
        
        # Create tools
        tools = [
            generate_tool(index, f, file_description=file_descriptions[f]["summary"]) 
            for f in files
        ]
        
        # Initialize LLM
        llm = OpenAI(model="gpt-4-turbo-preview")  # Updated model name
        
        # Initialize and run workflow
        workflow = RFPWorkflow(
            tools=tools,
            parser=parser,
            llm=llm,
            verbose=True,
            timeout=None,
        )
        
        rfp_template_path = Path(data_dir) / "jedi_cloud_rfp.pdf"
        if not rfp_template_path.exists():
            raise FileNotFoundError(f"RFP template not found at {rfp_template_path}")
        
        handler = workflow.run(rfp_template_path=str(rfp_template_path))
        async for event in handler.stream_events():
            if isinstance(event, LogEvent):
                if event.delta:
                    print(event.msg, end="", flush=True)
                else:
                    print(event.msg)
        
        response = await handler
        print(str(response))
        
    except Exception as e:
        _logger.error(f"Workflow failed: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 