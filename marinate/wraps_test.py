# Bad decorator - loses original function metadata
def bad_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

# Good decorator - preserves metadata using functools.wraps
from functools import wraps

def good_decorator(func):
    @wraps(func)  # This preserves the original function's metadata
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

# Example functions with decorators
@bad_decorator
def problematic_function(x):
    raise ValueError(f"Something went wrong with {x}")

@good_decorator
def better_function(x):
    raise ValueError(f"Something went wrong with {x}")

# Demonstration of the difference
print("Function names:")
print(f"Problematic function name: {problematic_function.__name__}")  # Prints 'wrapper'
print(f"Better function name: {better_function.__name__}")  # Prints 'better_function'

# When you run these functions, the traceback from better_function
# will point to the actual line in the function, while problematic_function
# will show the wrapper's location

# try:
#     problematic_function(42)
# except ValueError as e:
#     print("\nTraceback from problematic function:")
#     raise

try:
    better_function(42)
except ValueError as e:
    print("\nTraceback from better function:")
    raise