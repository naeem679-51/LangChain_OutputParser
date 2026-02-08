from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation' 
    )

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18,description='Age of the person')
    city: str = Field(description='Name of the city the person belong to')

parser = PydanticOutputParser(pydantic_object=Person) 

template= PromptTemplate(
    template='Genarate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'place':'Bangladeshi'})

result = model.invoke(prompt)

Final_result=parser.parse(result.content)

print(Final_result)