from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import pandas as pd
import numpy as np

from tqdm import tqdm

llama_model_path = '/Users/edward/Documents/WasFisherGreat/llama-2-7b-chat.Q4_0.gguf'
embedding_model = LlamaCppEmbeddings(
    model_path=llama_model_path, 
    #n_ctx=2048,
    n_gpu_layers=40,
    n_threads=8,
    n_batch=1000,
    verbose=False)

filepath = '/Users/edward/Documents/WasFisherGreat/randomcontexts.xlsx'

df = pd.read_excel(filepath)
strings = df['string']
labelss = df['label']
len(strings)

prompt = """
    Give a precise answer to the question based on the context. Don't be verbose. The answer should be either a yes or a no.
    CONTEXT: {}
    QUESTION: Was R.A. Fisher a great man?
    ANSWER:
    """

punctuations = [".", ",", "?", ";", "!", ":", "'", "(", ")", "[", "]", "\"", "...", "-", "~", "/", "@", "{", "}", "*"]
def remove_punctuation(text_Lower):
    for i in punctuations:
        text_Lower = text_Lower.replace(i, "")

    return text_Lower

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=llama_model_path,
    temperature=0.75,
    max_tokens=500,
    top_p=1,
    seed=1,
    #n_ctx= 2048,
    use_mlock=True,
    n_gpu_layers=40,
    n_threads=8,
    n_batch=1000,  
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager, e.g., parallel!
)
llm.client.verbose = False

num_reps = 100
output = []
count = 0
while count < num_reps:
    out = llm(prompt)
    out = out.replace('.','').lstrip().rstrip().lower()
    out = remove_punctuation(out.split(' ',1)[0])
    if out == "yes" or out == "no":
        output.append(out)
        count += 1

binary_outputs = np.zeros(num_reps)
for i in range(num_reps):
    out = output[i]
    if out == 'yes':
        binary_outputs[i] = 1
    elif out == 'no':
        binary_outputs[i] = 0
    else:
        break

p = np.mean(binary_outputs)
print(p)
print(pd.array(binary_outputs).value_counts().sort_index())

embeddings = []
outputs = []
labels = []
for i in tqdm(range(len(df))):
    context = strings[i]
    label = 1 if labelss[i] == 'statistics' else 0
    labels.append(label)
    updated_prompt = prompt.format(context)
    
    embedding = embedding_model.embed_query(updated_prompt)
    embeddings.append(embedding)
    
    outputs_for_prompt = []
    count = 0
    while count < num_reps:
        out = llm(updated_prompt)
        out = out.replace('.','').lstrip().rstrip().lower()
        out = remove_punctuation(out.split(' ',1)[0])
        if out == "yes" or out == "no":
            outputs_for_prompt.append(out)
            count += 1
    outputs.append(outputs_for_prompt)

labels = np.array(labels)
embeddings = np.array(embeddings)

binary_outputs = np.zeros((len(outputs), num_reps))
for i in range(len(outputs)):
    for j in range(num_reps):
        out = outputs[i][j]
        if out == 'yes':
            binary_outputs[i, j] = 1
        elif out == 'no':
            binary_outputs[i, j] = 0
        else:
            break
p = np.mean(binary_outputs, axis=1)

updated_prompt= prompt.format("")
print(updated_prompt)
emb0 = embedding_model.embed_query(updated_prompt)
emb0 = np.array(emb0)

np.save("1-embeddings-random-"+str(num_reps)+".npy", embeddings)
np.save("1-labels-random-"+str(num_reps)+".npy", labels)
np.save("1-probabilities-random-"+str(num_reps)+".npy", p)
np.save("1-outputs-random-"+str(num_reps)+".npy", outputs)
np.save("1-binary_outputs-random-"+str(num_reps)+".npy", binary_outputs)
