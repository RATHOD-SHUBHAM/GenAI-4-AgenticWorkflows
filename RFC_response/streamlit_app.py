import streamlit as st
import asyncio
from pathlib import Path
import tempfile
import json
from test_script import (
    setup_environment,
    initialize_parser,
    parse_and_generate_summaries,
    initialize_index,
    create_file_descriptions,
    generate_tool,
    RFPWorkflow,
)
from llama_index.llms.openai import OpenAI
import base64
import time  # Add this with other imports

st.set_page_config(page_title="RFP Analysis Tool", layout="wide")

# Add custom CSS for gradient title and layout
st.markdown("""
    <style>
    .gradient-text {
        background: linear-gradient(45deg, #FF4B4B, #7E3FF2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: bold;
    }
    .logo-img {
        padding: 10px;
        width: 250px;  /* Increased logo size */
        height: auto;  /* Maintain aspect ratio */
    }
    </style>
""", unsafe_allow_html=True)

async def process_rfp(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Setup environment and initialize components
        data_dir, data_out_dir, files = setup_environment()
        parser = initialize_parser()
        file_dicts = parse_and_generate_summaries(data_dir, data_out_dir, files, parser)
        index = initialize_index(files, file_dicts, parser)
        file_descriptions = create_file_descriptions(file_dicts)
        
        # Create tools
        tools = [
            generate_tool(index, f, file_description=file_descriptions[f]["summary"]) 
            for f in files
        ]
        
        # Initialize LLM
        llm = OpenAI(model="gpt-4-turbo-preview")
        
        # Initialize workflow
        workflow = RFPWorkflow(
            tools=tools,
            parser=parser,
            llm=llm,
            verbose=True,
            timeout=None,
        )
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        qa_placeholder = st.empty()
        
        # Run workflow
        handler = workflow.run(rfp_template_path=tmp_path)
        
        qa_pairs = []
        current_question = None
        
        async for event in handler.stream_events():
            if hasattr(event, 'msg'):  # Check if event has msg attribute
                if "Asked question:" in event.msg:
                    # Extract question and answer
                    parts = event.msg.split("\n")
                    question = parts[0].replace(">> Asked question: ", "")
                    answer = parts[1].replace(">> Got response: ", "")
                    qa_pairs.append({"question": question, "answer": answer})
                    
                    # Update the display
                    with qa_placeholder.container():
                        for qa in qa_pairs:
                            st.write("**Question:**", qa["question"])
                            st.write("**Answer:**", qa["answer"])
                            st.markdown("---")
                
                progress_placeholder.text(event.msg if not hasattr(event, 'delta') else "Processing...")
        
        response = await handler
        return qa_pairs, str(response)

    finally:
        # Cleanup temporary file
        Path(tmp_path).unlink()

def main():
    # Create columns for logo and title with specific ratios
    logo_col, space_col, title_col, end_col = st.columns([2, 0.5, 3, 1])
    
    # Logo in first column
    with logo_col:
        st.markdown(
            f"""
            <img src="data:image/png;base64,{base64.b64encode(open("/Users/shubhamrathod/PycharmProjects/RFC_response/logo.png", "rb").read()).decode()}" class="logo-img">
            """,
            unsafe_allow_html=True
        )
    
    # Title in third column (after spacing column)
    with title_col:
        st.markdown('<h1 class="gradient-text">RFP Analysis Tool</h1>', unsafe_allow_html=True)
    
    # Add spacing
    st.write("")
    
    # Subheader with icon using Streamlit's info container
    st.warning(
        "🤖 **AI-Powered RFP Analysis Tool**: Leverages intelligent AI agents to analyze RFPs and generate comprehensive responses with dynamic Q&A pairs.  🚀"
    )
    
    # Add spacing
    st.write("")
    st.write("")
    
    st.write("""
    Upload your RFP document (PDF) to analyze it and generate question-answer pairs.
    """)
    
    uploaded_file = st.file_uploader("Choose an RFP file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process RFP"):
            # Create a progress bar and container for fancy status
            progress_bar = st.progress(0)
            status_container = st.container()
            
            # Initialize status with fancy styling
            with status_container:
                st.markdown("""
                    <style>
                    .status-text {
                        background: linear-gradient(45deg, #1e3c72, #2a5298);
                        padding: 10px;
                        border-radius: 5px;
                        color: white;
                        font-weight: bold;
                        text-align: center;
                        margin: 10px 0;
                        animation: pulse 2s infinite;
                    }
                    @keyframes pulse {
                        0% { opacity: 1; }
                        50% { opacity: 0.7; }
                        100% { opacity: 1; }
                    }
                    </style>
                """, unsafe_allow_html=True)
            
            status_text = st.empty()
            
            # Initialize status
            status_text.markdown("<div class='status-text'>🚀 Analysing RFP Document...</div>", unsafe_allow_html=True)
            progress_bar.progress(10)
            
            # Run the async processing
            try:
                qa_pairs, final_response = asyncio.run(process_rfp(uploaded_file))
                
                # Update progress for different stages with emojis and fancy messages
                stages = [
                    ("📄 Parsing document and extracting content...", 30),
                    ("🔍 Analyzing document structure and content...", 50),
                    ("💡 Generating intelligent Q&A pairs...", 70),
                    ("✨ Preparing comprehensive analysis...", 90),
                    ("🎉 Processing complete! Finalizing results...", 100)
                ]
                
                for message, progress in stages:
                    status_text.markdown(f"<div class='status-text'>{message}</div>", unsafe_allow_html=True)
                    progress_bar.progress(progress)
                    time.sleep(1.5)  # Slightly longer delay to make the animation visible
                
                # Clear the progress indicators
                progress_bar.empty()
                status_text.empty()
                status_container.empty()
                
                # Display completion message with animation
                st.markdown("""
                    <style>
                    .success-message {
                        background: linear-gradient(45deg, #28a745, #20c997);
                        padding: 20px;
                        border-radius: 10px;
                        color: white;
                        font-weight: bold;
                        text-align: center;
                        margin: 20px 0;
                        animation: slideIn 1s ease-out;
                    }
                    @keyframes slideIn {
                        from {
                            transform: translateY(-20px);
                            opacity: 0;
                        }
                        to {
                            transform: translateY(0);
                            opacity: 1;
                        }
                    }
                    </style>
                    <div class='success-message'>
                        🎯 Analysis completed successfully! 
                        <br>
                        Ready to explore the insights
                    </div>
                """, unsafe_allow_html=True)
                
                # Display QA pairs in an expandable section
                with st.expander("View Question-Answer Pairs", expanded=True):
                    for qa in qa_pairs:
                        st.write("**Question:**", qa["question"])
                        st.write("**Answer:**", qa["answer"])
                        st.markdown("---")
                
                # Display the final markdown output
                with st.expander("View Final Analysis", expanded=True):
                    try:
                        with open("/Users/shubhamrathod/PycharmProjects/RFC_response/data_out_rfp/workflow_output/final_output.md", 'r') as f:
                            markdown_content = f.read()
                        st.markdown(markdown_content)
                    except Exception as e:
                        st.error(f"Error reading markdown file: {str(e)}")
                
                # Option to download results
                qa_json = json.dumps(qa_pairs, indent=2)
                st.download_button(
                    label="Download Q&A Pairs as JSON",
                    data=qa_json,
                    file_name="rfp_qa_pairs.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Error processing RFP: {str(e)}")

if __name__ == "__main__":
    main() 