import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import os
import csv
import itertools
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import AutoTokenizer, LlamaForCausalLM
import accelerate
from accelerate import disk_offload
from accelerate import init_empty_weights, infer_auto_device_map,disk_offload

#------------------------------------------------------------------------------------------
#Takes a prompt and returns the output with the hidden states
def generate_output(prompt):
    print(f"generating output for '{prompt}'")
    inputs = tokenizer(prompt,return_tensors="pt").to(device)
    
    with torch.no_grad():
      outputs = model(**inputs, output_hidden_states=True)
    
    return outputs
      
#-----------------------------------------------------------------------------------------
#Takes an output state and saves a .npy file of the hidden states
def generate_Plot(hidden_States):
    print("Saving hidden_states")
    hidden_states_cpu = [tensor.cpu().numpy() for tensor in hidden_States]
    np.save('tensors.npy', hidden_states_cpu)

#-----------------------------------------------------------------------------------------
#Takes a csv file and flips it across the diagonal
def mirror_list(data):
  data = [list(row) for row in data]
  for i in range(len(data)):
    for j in range(i+1, len(data[i])):
      data[i][j] = data[j][i]
      
  return data


#-----------------------------------------------------------------------------------------
#Writes the result to a csv file with a given name
def write_to_csv(data, filename):
  transposed_data = list(itertools.zip_longest(*data, fillvalue=''))
  filled_data = list(itertools.zip_longest(*transposed_data, fillvalue=''))
  mirrored_data = mirror_list(filled_data)
  with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    for line in mirrored_data:
      print(f"{line}")
      writer.writerow(line)
#-----------------------------------------------------------------------------------------
#Computes the Kullback-Leibler divergance between the two sets of tuples
def compute_jensen_shannon_divergence(tensor_tuple1, tensor_tuple2):
    # Convert tensors to probabilities using softmax
    prob1 = F.softmax(torch.stack(tensor_tuple1, dim=0), dim=0).squeeze().cpu().numpy()
    prob2 = F.softmax(torch.stack(tensor_tuple2, dim=0), dim=0).squeeze().cpu().numpy()

    # Ensure prob1 and prob2 have the same shape along the second dimension
    max_len = max(prob1.shape[1], prob2.shape[1])
    prob1_padded = np.pad(prob1, ((0, 0), (0, max_len - prob1.shape[1]), (0, 0)), mode='constant')
    prob2_padded = np.pad(prob2, ((0, 0), (0, max_len - prob2.shape[1]), (0, 0)), mode='constant')

    # Compute the average of the two distributions
    avg_prob = (prob1_padded + prob2_padded) / 2
    
    epsilon = 1e-10  # Small epsilon value to prevent dividing by 0
    prob1_padded += epsilon
    prob2_padded += epsilon
    avg_prob += epsilon

    # Compute KL divergences between each distribution and the average
    kl_div1 = np.sum(prob1_padded * np.log(prob1_padded / avg_prob))
    kl_div2 = np.sum(prob2_padded * np.log(prob2_padded / avg_prob))
    print(f"The kl_div1 is {kl_div1}")
    print(f"The kl_div2 is {kl_div2}")

    # Compute Jensen-Shannon Divergence as the average of the KL divergences
    js_divergence = (kl_div1 + kl_div2) / 2
    
    print(f"The js Divergence is {js_divergence}")

    return js_divergence

#-----------------------------------------------------------------------------------------
#compares the average point between two tensor tuples using an average point for both
def compare_AvgL2(tensor_tuple1, tensor_tuple2):
  
  
  #calculate the average point 
  avg_point1 = torch.mean(torch.stack(tensor_tuple1), dim=0)
  avg_point2 = torch.mean(torch.stack(tensor_tuple2), dim=0)
  
  max_size = [max(s1, s2) for s1, s2 in zip(avg_point1.shape, avg_point2.shape)]
    
  # Pad the average points to the maximum size
  padded_avg_point1 = torch.nn.functional.pad(avg_point1, (0, 0, 0, max_size[1] - avg_point1.shape[1], 0, max_size[2] - avg_point1.shape[2]))
  padded_avg_point2 = torch.nn.functional.pad(avg_point2, (0, 0, 0, max_size[1] - avg_point2.shape[1], 0, max_size[2] - avg_point2.shape[2]))
  
  
  l2_norm = torch.norm(padded_avg_point1 - padded_avg_point2, p=2)
  
  return l2_norm

#-----------------------------------------------------------------------------------------
#compares two sets of tuples point by point with an L2 norm
def compare_P2P(tensor_tuple1, tensor_tuple2):
  print("comparing layers")
  mse_values = []
  mse_loss = nn.MSELoss()
  
  #get the shape of the largest tensor
  max_shape = torch.Size(max(tensor1.shape for tensor1 in tensor_tuple1 + tensor_tuple2))
  
  #pad the tensors to match the largest one
  tensor_tuple1_padded = [torch.nn.functional.pad(tensor1, (0, *(d2 - d1 for d1, d2 in zip(tensor1.shape, max_shape)))) for tensor1 in tensor_tuple1]
  tensor_tuple2_padded = [torch.nn.functional.pad(tensor2, (0, *(d2 - d1 for d1, d2 in zip(tensor2.shape, max_shape)))) for tensor2 in tensor_tuple2]
  
  for i in range(len(tensor_tuple1_padded)):
    
    mse = mse_loss(tensor_tuple1_padded[i], tensor_tuple2_padded[i])
    mse_values.append(mse)
  
  avg_mse = sum(mse_values)
  
  return avg_mse

#-----------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
cache_dir = "/mnt/beegfs/cwkavana/Cache_dir"



new_adjectives = ["Bold", "Dynamic","Energetic","Audacious","Exuberant"]


#end of init

#Loads the model and the tokenizer
print("Trying model")

#model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",cache_dir=cache_dir,device_map='auto')
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",cache_dir=cache_dir,device_map='auto')

print("finished model")


print("Trying tokenizer")
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=cache_dir)
print("tokenizer done")

responses = []

prompts_file = "/home/cwkavana/591/Llama_project/Prompts"
output_folder = "/home/cwkavana/591/Llama_project/outputs/"
index = 0

for exp_file in os.listdir(prompts_file):
  output_matrix = []
  prompt_file = os.path.join(prompts_file, exp_file)
  index = index + 1
  output_path = os.path.join(output_folder, f"output{index}JSD.txt")
  matrix_output_path = os.path.join(output_folder, f"output_matrix{index}JSD.csv")

  #print the prompt list from external file
  with open(prompt_file, 'r') as file:
    prompts = [line.strip() for line in file.readlines()]
    print("loading prompts:")
    for prompt in prompts:
      print(prompt)
  
  #check if output file location exists. If it doesnt, make it
  if os.path.exists(output_path):
    print("output exists.")
  else:
    # Create a new file at the specified location
    with open(output_path, 'w') as new_file:
        # You can write initial content to the file if needed
        new_file.write("")
    print("Outputfile created at", output_path)
    
    
  with open(output_path, "w") as file:
    control_prompt = prompts[0] #first prompt is the control
    print(f"control prompt is: {control_prompt}")
    output = generate_output(control_prompt) #first output
    hidden_states_holder = output.hidden_states #hidden states
    tensor_list = list(hidden_states_holder)#tensors
      
    file.write(f"control prompt: {control_prompt}\n") #change to first prompt
    output_ids = output.logits.argmax(dim=-1) #text from base output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    file.write(f"Generated text: {output_text}\n")
      
    #for hidden_state in tensor_list:
      #file.write(str(hidden_state.size()))
    
    #generate plot
    generate_Plot(hidden_states_holder)
    
    file.write("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n") #formatting for output file
    #Code for itterating through all prompts
    
    #print(f"Output for origonal version:\n {output}\n")
    new_hidden_states_holder = [] #an array of hidden layer outputs
    for prompt in prompts:
      row = []
      new_output = generate_output(prompt) #new output
      print(f"New prompt: {prompt}\n")
      file.write(f"New prompt: {prompt}\n")
      
      #get and print out the text of the output
      new_output_ids = new_output.logits.argmax(dim=-1)
      new_output_text = tokenizer.decode(new_output_ids[0], skip_special_tokens=True)
      print(new_output_text)
      file.write(str(f"the new generated text was:\n {new_output_text}\n"))
      
      #get and print out the norm for the output:
      new_hidden_states_holder.append(new_output.hidden_states) #adds the output newly generated to the tuple
      current_hidden = new_hidden_states_holder[len(new_hidden_states_holder) - 1] #the newest hidden output
      #compare tuples of the newest hidden state and the base one.
      #MSE = compare_AvgL2(hidden_states_holder, current_hidden)
      #MSE = compare_P2P(hidden_states_holder, current_hidden)
      MSE = compute_jensen_shannon_divergence(hidden_states_holder, current_hidden)
      print(f"first MSE: {MSE}\n")
      file.write(str(f"L2 norm from base prompt was: {MSE}\n"))
      #Do process for each other output as well
      for j, held_state in enumerate(new_hidden_states_holder):
        #MSE = compare_AvgL2(held_state, current_hidden)
        #MSE = compare_P2P(held_state, current_hidden)
        MSE = compute_jensen_shannon_divergence(held_state, current_hidden)
        row.append(f"{MSE}")
        print(f"new MSE: {MSE}\n")
        file.write(str(f"L2 norm from prompt {j} was: {MSE}\n"))
      output_matrix.append(row)
      #print the size of each tensor
      new_tensor_list = list(current_hidden) 
          
      
      #formatting for output file
      file.write("\n ================================================================================================== \n")
      file.write("\n")
    
    
  #create a file in the output directory that contains a matrix of the difference value for each prompt  
  
  print(str(output_matrix))
  write_to_csv(output_matrix, matrix_output_path)