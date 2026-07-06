from typing           import Callable, Any, Dict, List, Union
from docstring_parser import parse

import inspect
import os
import base64

import random
import string


def get_function_details(func: Callable) -> Dict[str, Any]:
    '''Inspects a function and returns its parameters and docstring in a structured format.

    Args:
        func (Callable): The target function to be inspected.

    Returns:
        Dict[str, Any]: A dictionary containing 'parameters' and 'docstring' keys.
            'parameters' is a list of dictionaries, each describing a parameter.
            'docstring' is the docstring of the function.

    Raises:
        TypeError: If the provided object is not callable.
    '''
    if not callable(func):
        raise TypeError('The provided object is not callable. Please provide a valid function or callable object.')

    docstring = inspect.getdoc(func) or 'No docstring provided.'
    
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):
        return {
            'parameters': 'Could not retrieve parameters for this function.',
            'docstring': docstring
        }

    parameters_details: List[Dict[str, Any]] = []
    for name, param in signature.parameters.items():
        param_info = {
            'name': name,
            'kind': str(param.kind.description),  # e.g., 'positional or keyword'
            'default': param.default if param.default is not inspect.Parameter.empty else 'N/A',
            'annotation': param.annotation.__name__ if hasattr(param.annotation, '__name__') \
                          and param.annotation is not inspect.Parameter.empty else 'N/A'
        }
        parameters_details.append(param_info)
        
    return {
        'parameters': parameters_details,
        'docstring': docstring
    }


def analyze_tool_function(func: Callable) -> Dict[str, Any]:
    '''Analyzes a function in detail, combining its signature and docstring parser results.

    Retrieves detailed information for each parameter. Can automatically identify
    multiple docstring styles such as Google, reST, NumPy, etc.

    Args:
        func (Callable): The target function to analyze.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'docstring': The combined summary description of the function.
            - 'parameters': A list of enhanced parameter dictionaries containing names,
              kinds, defaults, annotations, descriptions, and requirement status.
    '''
    basic_details = get_function_details(func)
    original_docstring = basic_details.get('docstring', '')
    
    parsed_docstring = parse(original_docstring)
    
    param_descriptions = {
        param.arg_name: param.description for param in parsed_docstring.params
    }
    
    summary_parts = []
    if parsed_docstring.short_description:
        summary_parts.append(parsed_docstring.short_description)
    if parsed_docstring.long_description:
        summary_parts.append(parsed_docstring.long_description)
    summary = '\n\n'.join(summary_parts) if summary_parts else ''
    
    enhanced_parameters = []
    if isinstance(basic_details['parameters'], list):
        for param in basic_details['parameters']:
            param_name = param['name']
            
            enhanced_param = param.copy()
            
            enhanced_param['description'] = param_descriptions.get(
                param_name, 'No description found in docstring.'
                ).replace('\n', ' ')
            
            enhanced_param['required'] = (param['default'] == 'N/A')
            
            enhanced_parameters.append(enhanced_param)
            
    return {
        'docstring': summary,
        'parameters': enhanced_parameters
    }


def safecode(length: int = 4, exclude_confusing: bool = False) -> str:
    '''Generates a random safe code consisting of letters and digits.

    Args:
        length (int): The length of the code to generate. Defaults to 4.
        exclude_confusing (bool): If True, excludes confusing characters
            ('0oO1iIlLq9g') to reduce human error. Defaults to False.

    Returns:
        str: The generated random code.
    '''
    characters = string.ascii_letters + string.digits

    if exclude_confusing:
        confusing_chars = '0oO1iIlLq9g'
        characters = ''.join(c for c in characters if c not in confusing_chars)

    code = ''.join(random.choices(characters, k=length))
    return code


def req_base64_file(path: str) -> str:
    '''Reads a file and returns its content encoded as a Base64 string.

    Args:
        path (str): The path to the file.

    Returns:
        str: The Base64 encoded string of the file content.
    '''
    return base64.b64encode(req_file(path, mode='rb')).decode('utf-8')


def req_file(path: str, mode: str = 'r', encoding: str = 'utf-8') -> Union[str, bytes]:
    '''Reads and returns the content of a file.

    Args:
        path (str): The path to the file.
        mode (str): The mode in which the file is opened (e.g., 'r', 'rb'). Defaults to 'r'.
        encoding (str): The encoding used to decode the file when reading in text mode. Defaults to 'utf-8'.

    Returns:
        Union[str, bytes]: The content of the file. Returns an empty string or empty bytes if the file does not exist.
    '''
    if not os.path.isfile(path): return b'' if 'b' in mode else ''
    if 'b' in mode:
        with open(path, mode) as f:
            return f.read()
    else:
        with open(path, mode, encoding=encoding) as f:
            return f.read()
