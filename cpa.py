from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from IPython.display import display
from dataclasses import dataclass
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import pandas as pd
import anthropic
import base64
import json
import os

class DataFrameOperation(BaseModel):
    """Call a member function of a Pandas DataFrame in the "stack". 
    
    -   All DataFrame functions are valid, but only functions.
        So for example, columns must be accessed with __getitem__(key="column_name") instead of [] and getting a list of the 
        columns requires a call to keys() instead of the usual columns.

    -   To call a function of an attribute, provide the attribute and function name with a . separator, as in `plot.bar(...)`

    The operation returns its result, which will be presented as its string representation, i.e. calling __repr__.
    If the result is a DataFrame (including a groupby) it will also be pushed on to the stack, if it's a Pandas Series it will be loaded into the 
    series register (replacing whatever was in the register). If the operation returns a chart, the base64 image will be returned.
    Note that because the operation only saves Series and DataFrames, it is futile to call functions like groupby that 
    retunrn other intermediate objects that won't be saved. Please don't call such functions.
    """
    target_frame: int = Field(description="""The position of the target dataframe in the stack on which to operate, with 0 being the top.""")

    function: str = Field(description="""
    A valid Pandas DataFrame member function, as in a callable present in `dir(df)` where df is a Pandas dataframe.
    """)
    
    kwargs: Dict = Field(description="""
    Key-value pairs corresponding to all the arguments to call the function with. For a function with no arguments
    this is an empty dict. The values must be integers, strings or lists of the same, they cannot be functions or expressions.
    """)

class SeriesOperation(BaseModel):
    """Call a member function of a Pandas Series in the series register. All Series functions are valid, but only functions.
    So for example, it must be indexed with __getitem__ instead of [].

    The operation returns its result, which will be presented as its string representation, i.e. calling __repr__.
    If it's a pandas series is will be loaded into the series register (replacing whatever was in the register). 
    If the operation returns a chart, the base64 image will be returned. 
    """

    function: str = Field(description="""
    A valid Pandas Series member function, as in a callable present in `dir(df)` where df is a Pandas dataframe
    """)
    
    kwargs: Dict = Field(description="""
    Key value pairs corresponding to all the arguments to call the function with. For a function with no arguments
    this is an empty dict. The values must be integers, strings or lists of the same, they cannot be functions or expressions.
    """)

class Pop(BaseModel):
    """Pop the top DataFrame from the stack, removing it from memory. Returns the __repr__ of this DataFrame"""

class SeriesAssign(BaseModel):
    """Assign the series in the series register to the dataframe at the top of the stack. This is to be used in place
    of assignment operations that are not possible with the available operations.
    For example, df.assign(c=df["a"] + df["b"]) is not possbile by single function calling. Instead we could call eval("a + b")
    which puts a series equal to the sum of columns a and b in the series register, then series_assign(column_name="c") to assign
    to a column called c in the dataframe.
    """
    column_name: str = Field(description="""
    The name of the column where the assigned series will be added to the dataframe
    """)

    in_place: Optional[bool] = Field(default=False, description="""
    If true the series is assigned to the dataframe on top of the stack an the stack.
    If false, a new dataframe is pushed onto the stack
    """)

tools = {"pop":Pop,
         "dataframe_operation":DataFrameOperation,
         "series_operation":SeriesOperation,
         "series_assign":SeriesAssign}

tool_schemas = []
for name, cls in tools.items():
    schema = json.loads(cls.schema_json())
    for _, d in schema["properties"].items():
        d.pop("title")

    _tool_schema = { "name":name,
                          "description": schema["description"],
                          "input_schema" : {
                              "type":"object",
                              "properties":schema["properties"],
                              #"required":schema["required"]
                          }
    }

    if "required" in schema:
        _tool_schema["input_schema"]["required"] = schema["required"]

    tool_schemas.append(_tool_schema)

@dataclass
class State:
    messages : List
    stack : List
    series : pd.Series
    steps : List
    tool_call : bool

_system = """You are acting as a data analysis agent, working with python's Pandas package to fullfill a user request."""

_preamble = """Your starting point is a Pandas DataFrame containing the data set to be analyzed to fullfill the request. 
You can call Pandas functions as described in the tools in order to perform your analysis. 

You have access to a stack of DataFrames, where the initial element is the data set to be analyzed. You also have a single
register that can store a Pandas series and is overwritten when a new one is returned.
When you don't need intermediate results at the top of the stack, please pop them off to keep the size manageable.
When you have determined the final answer to the user request, or are stuck and cannot go further, please provide your
final reply without using a tool.
"""

def init(user_request : str, csv_file : str):

    df = pd.read_csv(csv_file)
    stack = [df]
    stack_repr = "\n\n".join([f"<stack element {n}>\n{df.__repr__()}\n</stack element {n}>" for n,df in enumerate(stack)][::-1])
    series = None
    steps = []

    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": _preamble + f"""
                    User request: {json.dumps(user_request)}
                    Stack and Register
                    <stack>
                    {stack_repr}
                    </stack>
                    <series register>
                    {series.__repr__()}
                    </series register>
                    """
                },

         
    ] }]

    return State(   messages=messages,
                    stack=stack,
                    series=series,
                    steps=steps,
                    tool_call=True,
                    )


def _resolve(base, function):
    path = function.split(".")
    new_base = getattr(base, path[0])
    if len(path) == 1:
        return new_base 
    return _resolve(new_base, ".".join(path[1:]))

def step(state):
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2048,
        temperature=0,
        system=_system,
        tools = tool_schemas,
        tool_choice = {"type": "auto", "disable_parallel_tool_use":True,},
        messages=state.messages
    )

    #this happens regardless
    response_dict= json.loads(message.json())
    state.messages.append({"role":"assistant", "content":response_dict["content"]})
    
    msg_text = ""
    tool_input = ""
    tool_name = ""
    state.tool_call = False
    step_texts = []
    
    
    for c in message.content:
        if type(c) == anthropic.types.text_block.TextBlock:
            msg_text = (c.text)
            step_texts.append(c.text)
            print(msg_text)
            
        if type(c) == anthropic.types.tool_use_block.ToolUseBlock:
            state.tool_call = True
            tool_name = c.name
            tool_input = c.input
            tool_id = c.id
            step_texts.append(tool_name + ": " + json.dumps(tool_input))
            print(tool_name)
            print(tool_input)

    _res = None
 
    if tool_name:
        if tool_name == "pop":
            res = state.stack.pop().__repr__()

        elif tool_name == "series_assign":
            if "in_place" in tool_input and not tool_input["in_place"]:
                state.stack.append(state.stack[-1].copy())
                
            state.stack[-1][tool_input["column_name"]] = state.series
            res = state.stack[-1].__repr__()
        
        elif tool_name == "dataframe_operation":
            #if tool_input["function"] == "groupby":
            #    _res = None
            #    res = "Error: groupby cannot be used here, it does not return a dataframe it returns a grou
            try:
                _res = _resolve(state.stack[tool_input["target_frame"]], tool_input["function"])(**tool_input["kwargs"])
                #_res = getattr(state.stack[tool_input["target_frame"]], tool_input["function"])(**tool_input["kwargs"])
                res = _res.__repr__()
            except Exception as e:
                res = e.__repr__()
                step_texts.append(res)
                print(res)
                
                
        elif tool_name == "series_operation":
            try:
                #_res = getattr(state.series, tool_input["function"])(**tool_input["kwargs"])
                _res = _resolve(state.series, tool_input["function"])(**tool_input["kwargs"])
                res = _res.__repr__()
            except Exception as e:
                res = e.__repr__()
                step_texts.append(res)
                print(res)
            
        
        if isinstance(_res, pd.DataFrame) or isinstance(_res,pd.api.typing.DataFrameGroupBy):
                state.stack.append(_res)
                step_texts.append("Dataframe\n"+_res.head().__repr__()+"...")
                print("Dataframe")
                display(_res.head(5))
                print("...")
        elif isinstance(_res, pd.Series):
                state.series = _res
                step_texts.append("Series\n"+_res.head().__repr__()+"...")    
                print("Series:")
                display(_res.head())
                print("...")

        #only happens with a tool call
        stack_repr = "\n\n".join([f"<stack element {n}>\n{df.__repr__()}\n</stack element {n}>" for n,df in enumerate(state.stack)][::-1])
        _state =  f"""
        
        Stack and Register
                    <stack>
                    {stack_repr}
                    </stack>
                    <series register>
                    {state.series.__repr__()}
                    </series register>"""

        step_data = {"text":"\n\n".join(step_texts)}
        
        msg_content = []

        if plt.get_fignums():
            # encode as base64
            output = BytesIO()
            plt.savefig(output, format='png')
            plt.show()
            im_data = output.getvalue()
            image_data = base64.b64encode(im_data).decode("utf-8")

            step_data["image"] = image_data
            
            msg_content.append( {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                },)
        
        msg_content.append({"type":"text", "text":res + _state})
        tool_response =  [{
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": msg_content,#res + _state
                }]

        
        
        
        state.messages.append({"role":"user", "content":tool_response})
        state.steps.append(step_data)
    

