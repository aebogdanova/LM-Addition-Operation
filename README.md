## Fine-tuning Language Model for Addition Operation Task

### Description
The repository includes a dataset and scripts for fine-tuning the GPT-2 model to perform the addition operation task, based on the methodology described in the article ["Toolformer: Language Models Can Teach Themselves to Use Tools"](https://arxiv.org/abs/2302.04761). The technical report is available [here](/Report.pdf).

### Model

To run fine-tuning:
```
python train.py
```

### Evaluation 
The fine-tuned model is available [here](https://disk.yandex.ru/d/Q2fLuMmlVFRC2w). 
Please download it and place to `./model` directory.

To see results:
```
python evaluate.py
```

### Results
See accuracy scores in `results/results_average.json`.