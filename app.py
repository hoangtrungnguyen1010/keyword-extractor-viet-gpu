import gradio as gr
import torch
import os

from pipeline import KeywordExtractorPipeline

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def extract_keyword(title, text, top_n, ngram_low_range, ngram_high_range, min_freq, diversify_result):
    inp = {"text": text, "title": title}
    keyword_ls = kw_pipeline(inputs=inp, min_freq=min_freq, ngram_n=(ngram_low_range, ngram_high_range),
                             top_n=top_n, diversify_result=diversify_result)
    result = ''
    for kw, score in keyword_ls:
        result += f'{kw}: {score}\n'
    return result


if gr.NO_RELOAD:
    print("Loading PhoBERT model")
    phobert = torch.load(f'{DIR_PATH}/pretrained-models/phobert.pt')
    phobert.eval()

    print("Loading NER model")
    ner_model = torch.load(f'{DIR_PATH}/pretrained-models/ner-vietnamese-electra-base.pt')
    ner_model.eval()
    kw_pipeline = KeywordExtractorPipeline(phobert, ner_model)

if __name__ == "__main__":
    demo = gr.Interface(fn=extract_keyword,
                        inputs=[
                            gr.Text(
                                label="Title",
                                lines=1,
                                value="Enter title here",
                            ),
                            gr.Textbox(
                                label="Text",
                                lines=5,
                                value="Enter text here",
                            ),
                            gr.Number(
                                label="Top N keywords",
                                info="Number of keywords retrieved",
                                value=10
                            ),
                            gr.Number(
                                label="Ngram low range",
                                value=1
                            ),
                            gr.Number(
                                label="Ngram high range",
                                value=3
                            ),
                            gr.Number(
                                label="Ngram minimum frequency",
                                value=1
                            ),
                            gr.Checkbox(
                                label="Diversify result"
                            )
                        ],
                        # inputs=["text", "textbox", "number", "number", "number", "number", "checkbox"],
                        outputs=gr.Textbox(
                            label="Keywords Extracted",
                        )
                        )

    demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
