import gradio as gr
from theme_classifier import ThemeClassifier

def get_themes(theme_list_str,subtitles_path,save_path):
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path,save_path)

    # Remove dialogue from the theme list
    output_df = output_df.drop('dialogue', axis=1)

    theme_output = output_df.drop(['episode', 'script'], axis=1).sum().reset_index()
    theme_output.columns = ['theme', 'score']

    # print(theme_output)

    # output_df = output_df[theme_list].sum().reset_index()
    # output_df.columns = ['Theme','Score']
    # print(output_df)

    output_chart = gr.BarPlot(
        theme_output,
        x="Theme",
        y="Score",
        title="Series Themes",
        tooltip=["Theme","Score"],
        vertical=False,
        width=500,
        height=260
    )

    return output_chart

def main():
    with gr.Blocks() as iface:
        # Theme Classification Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Claasifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(label="Subtitles or script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button =gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list,subtitles_path,save_path], outputs=[plot])

    iface.launch(share=True)


if __name__ == '__main__':
    # get_themes('friendship,hope,sacrifice,battle,self development,betrayal,love,dialogue',
    #             'data/subtitles',
    #             'stubs/output_file.csv')
    main()
