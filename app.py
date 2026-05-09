import gradio as gr
import os
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. API Keys Securely Fetch Karna
groq_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# 2. AI Brain (Groq) aur Search Tool ka Setup
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key)
search_tool = TavilySearchResults(max_results=3, tavily_api_key=tavily_key)

# 3. Asli Agentic Workflow (Main Logic)
def process_cv_and_search(file):
    if not groq_key or not tavily_key:
        return "0", "Error", "Failed", "⚠️ Error: API Keys set nahi hain. Settings check karein."
    if file is None:
        return "0", "Waiting...", "Standby", "⚠️ Please ek PDF CV upload karein."

    try:
        # Step A: CV Parhna
        reader = PdfReader(file.name)
        cv_text = "".join([page.extract_text() for page in reader.pages])
        
        # Step B: Agent khud sochega (Plan)
        plan_prompt = f"Analyze this CV. Extract primary job title and top 3 skills. Create a single, effective job search query. CV: {cv_text}"
        search_query = llm.invoke(plan_prompt).content.strip()
        
        # Step C: Real-time internet par jobs search karna
        jobs = search_tool.invoke(search_query + " recent job openings Pakistan")
        
        # Format fix for Tavily
        if isinstance(jobs, str):
            jobs = [{'content': jobs, 'url': '#'}]
            
        jobs_found_count = str(len(jobs))
        
        # Step D: Har job ko CV ke sath compare karna (Evaluate & Explain)
        final_output = f"### 🔍 Agent Search Query\n**`{search_query}`**\n\n---\n\n"
        
        for i, job in enumerate(jobs):
            if isinstance(job, dict):
                job_desc = job.get('content', str(job))
                job_link = job.get('url', '#')
            else:
                job_desc = str(job)
                job_link = '#'
                
            eval_prompt = f"You are an expert HR. Compare this CV with the Job Description. Give a Match Score (out of 100%) and write a short, crisp paragraph explaining WHY this job is a strong or weak match. CV: {cv_text}\nJob: {job_desc}"
            evaluation = llm.invoke(eval_prompt).content
            
            final_output += f"### 🏢 Job Match {i+1}: [Click to Apply]({job_link})\n{evaluation}\n\n---\n"
            
        # Returning Stats for the Dashboard Cards + Final Output
        return jobs_found_count, "Just Now", "Active", final_output
        
    except Exception as e:
        return "0", "Error", "Failed", f"🚨 An error occurred: {str(e)}"

# 4. CUSTOM THEME (Dark Mode + Neon Green Nano Banana Style)
custom_css = """
body, .gradio-container { background-color: #0b0f0e !important; color: #e0e0e0 !important; }
.gr-button-primary { background-color: #10B981 !important; color: #000 !important; border: none !important; font-weight: bold; }
.gr-button-secondary { background-color: #1a3d28 !important; color: #10B981 !important; border: 1px solid #10B981 !important; }
.gr-panel { background-color: #0f1714 !important; border: 1px solid #1a3d28 !important; border-radius: 8px; }
h1, h2, h3, h4 { color: #10B981 !important; }
.stat-box { background-color: #0b0f0e; border: 1px solid #10B981; padding: 15px; border-radius: 8px; text-align: center; }
.stat-value { font-size: 24px; color: #10B981; font-weight: bold; }
.sidebar-nav { padding: 10px 0; color: #a3b8aa; font-weight: bold; cursor: pointer; }
.sidebar-nav:hover { color: #10B981; }
"""

# 5. DASHBOARD UI STRUCTURE
with gr.Blocks(css=custom_css, theme=gr.themes.Monochrome()) as demo:
    
    # Top Header
    with gr.Row():
        gr.HTML("<h2 style='margin-bottom: 0;'>⚙️ JobScout AI :: Career Agent Dashboard</h2>")
    
    with gr.Row():
        # --- LEFT SIDEBAR (Mimicking the image's left menu) ---
        with gr.Column(scale=1, min_width=200):
            gr.HTML("""
            <div style='background-color: #0f1714; padding: 20px; border-radius: 8px; border: 1px solid #1a3d28; height: 100%;'>
                <h3 style='color: #10B981; margin-top: 0;'>Navigation</h3>
                <div class='sidebar-nav'>🏠 Dashboard</div>
                <div class='sidebar-nav'>🤖 Agent Settings</div>
                <div class='sidebar-nav'>📊 Match Reports</div>
                <hr style='border-color: #1a3d28; margin: 20px 0;'>
                <p style='color: #a3b8aa; font-size: 12px;'>Upload your CV here to trigger the autonomous agent.</p>
            </div>
            """)
            cv_input = gr.File(label="Upload Profile (PDF)", file_types=[".pdf"])
            submit_btn = gr.Button("🚀 Trigger Agent", variant="primary")
        
        # --- RIGHT MAIN CONTENT (Live Analytics + Results) ---
        with gr.Column(scale=4):
            gr.HTML("<h4>Live Analytics</h4>")
            
            # 3 Stat Cards (Mimicking Total Scraped, Latest Time, Active Jobs)
            with gr.Row():
                stat_jobs = gr.Textbox(label="Active Jobs Matched", value="0", lines=1, max_lines=1)
                stat_time = gr.Textbox(label="Latest Agent Run", value="Waiting...", lines=1, max_lines=1)
                stat_status = gr.Textbox(label="Agent Status", value="Standby", lines=1, max_lines=1)
            
            gr.HTML("<h4 style='margin-top: 20px;'>Agentic Evaluation & Market Overview</h4>")
            
            # Main Results Box
            output_text = gr.Markdown(label="Ranked Matches & Reasoning", value="*Awaiting CV upload... Agent will autonomously ingest, plan, and evaluate matches here.*")
    
    # Connecting the button to the function and mapping outputs to the stat cards + markdown
    submit_btn.click(
        fn=process_cv_and_search, 
        inputs=cv_input, 
        outputs=[stat_jobs, stat_time, stat_status, output_text]
    )

demo.launch()