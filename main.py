import vector_db
import encode
import llm_generation

while(True):
  print("Welcome! \n")
  query = input("What's on your mind? ") # "why should recreational marijuana be illegal" (query id 6)
  print("Query: " + query + "\n")
  # get query embedding from encoder
  q_emb = encode.encode(query)
  # pass embedding to search vector db for top k results
  texts = vector_db.search(q_emb, 3)
  print("\n")
  # build prompt with documents as context for LLM
  prompt = llm_generation.build_prompt(query, texts)
  # generate NLP response to prompt
  result = llm_generation.generate(prompt)
  print("\nLLM Answer: ")
  print(result)

  repeat = input("\nAnything else? (quit to quit)")
  if(repeat == "quit"):
    print("See you next time! Bye!")
    break