import re 
import random 

def main():
    with open("./data/prompts_dataset.txt", "r", encoding="utf-8") as prompts_file:
        prompts = prompts_file.readlines()

    with open("./data/train.txt", "a", encoding="utf-8") as train_file:
      for i in range(500):
          for prompt in prompts:
              num_1 = random.randint(1e+9, 1e+19)
              num_2 = random.randint(1e+9, 1e+19)
              prompt = re.sub("<NUM1>", str(num_1), prompt)
              prompt = re.sub("<NUM2>", str(num_2), prompt)
              prompt = re.sub("<SUM>", str(num_1 + num_2), prompt)
              train_file.write(prompt)

if __name__ == "__main__":
    main()
