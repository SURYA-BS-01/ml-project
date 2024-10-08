from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()

from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
from text_classification import JutsuClassifier

from character_chatbot import CharacterChatBot

app = FastAPI()

# Mount a static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get():
    html_file = Path("index.html")
    return HTMLResponse(content=html_file.read_text(), status_code=200)

@app.post("/character_chatbot")
async def chat_with_character_chatbot(message: str = Form(...), history: str = Form(...)):
    try:
        # Token is fetched from environment variables
        character_chatbot = CharacterChatBot("SuryaBS/Naruto_Llama-3-8B",
                                             huggingface_token=os.getenv("huggingface_token"))

        # Split history by commas to convert it back into a list format
        history_list = history.split(',') if history else []
        output = character_chatbot.chat(message, history_list)
        output = output['content'].strip()

        return output
    except Exception as e:
        return {"error": str(e)}


@app.post("/generate-bar-chart/")
async def generate_bar_chart(theme_list: str = Form(...), subtitles_path: str = Form(...), save_path: str = Form(...)):
    try:
        theme_list = theme_list.split(',')
        theme_classifier = ThemeClassifier(theme_list)
        output_df = theme_classifier.get_themes(subtitles_path, save_path)

        # Remove dialogue from the theme list
        output_df = output_df.drop('dialogue', axis=1)

        theme_output = output_df.drop(['episode', 'script'], axis=1).sum().reset_index()
        theme_output.columns = ['theme', 'score']

        # Plot bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(theme_output['theme'], theme_output['score'], color='skyblue')
        plt.xlabel('Theme')
        plt.ylabel('Score')
        plt.title('Theme Scores')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot as an image in the static folder
        static_folder = Path("static")
        static_folder.mkdir(exist_ok=True)
        chart_path = static_folder / 'theme_bar_chart.png'
        plt.savefig(chart_path)
        plt.close()

        return {"image_url": "/static/theme_bar_chart.png"}
    except Exception as e:
        print(f"Error generating chart: {str(e)}")
        return {"error": str(e)}

@app.get("/character-network")
async def get_character_network(subtitles_path: str, ner_path: str):
    try:
        ner = NamedEntityRecognizer()
        ner_df = ner.get_ners(subtitles_path, ner_path)

        character_network_generator = CharacterNetworkGenerator()
        relationship_df = character_network_generator.generate_character_network(ner_df)
        html = character_network_generator.draw_network_graph(relationship_df)

        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        return {"error": str(e)}


@app.get("/themes-bar-chart")
async def get_bar_chart_image():
    chart_path = Path("static/theme_bar_chart.png")
    if chart_path.exists():
        return FileResponse(chart_path)
    else:
        return {"error": "Chart not found"}

@app.post("/text-classification/")
async def classify_text(
    model_path: str = Form(...),
    data_path: str = Form(...),
    text_input: str = Form(...)
):
    try:
        jutsu_classifier = JutsuClassifier(model_path = model_path,
                                       data_path = data_path,
                                       huggingface_token = os.getenv('huggingface_token'))
    
        output = jutsu_classifier.classify_jutsu(text_input)
        output = output[0]
        
        return {"jutsu_type": output}
    
    except Exception as e:
        return {"error": str(e)}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)