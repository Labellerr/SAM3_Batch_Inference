from sam3_batch_inference import run_batch_inference


def main():
    INPUT_FOLDER = "flower_sample_img"
    MODEL_CHECKPOINT = r"model\sam3.pt"
    TEXT_PROMPTS = ["Red Flower", 
                    "Yellow Flower", 
                    "White Flower",
                    "Violet Flower",
                    ]
    run_batch_inference(INPUT_FOLDER,
                        TEXT_PROMPTS,
                        MODEL_CHECKPOINT, 
                        )

if __name__ == "__main__":
    main()

