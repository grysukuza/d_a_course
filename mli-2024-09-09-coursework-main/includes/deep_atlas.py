# Standard Deep Atlas set-up cell. Do not edit.
# Run before starting work on the rest of the exercise below.

import sys, os, datetime
from IPython import get_ipython
import json

FILL_THIS_IN = "THIS_VARIABLE_STILL_NEEDS_TO_BE_FILLED_IN" # The FILL_THIS_IN variable will be used to guide you

environment = None
environment_initialized = False

environment_exception_message_template = '''{environment_name} is not yet supported for this Deep Atlas exercise.

To run on {environment_name} anyway:
- Pass `allow_{environment_token}=True` when calling `initialize_exercise()`
- Be prepared for problems
'''
colab_exception_message = environment_exception_message_template.format(environment_name='Google Colab', environment_token='colab')
kaggle_exception_message = environment_exception_message_template.format(environment_name='Kaggle', environment_token='kaggle')

unrecognized_environment_exception_message = '''Unrecognized Environment!

If you're attempting to run this notebook locally:
- Set up your Pipenv virtual environment
- Restart your editor
- Select the appropriate kernel
- Rerun this block.

To run on an unrecognized environment anyway:
- Pass `allow_unrecognized=True` when calling `initialize_exercise()`
- Be prepared for big problems!
'''

def initialize_environment(allow_colab=True, allow_kaggle=True, allow_unrecognized=False):
    global environment, environment_initialized

    if 'virtualenvs' in sys.executable:
        environment = 'VIRTUAL'
        print("🎉 Running in a Virtual environment")
    elif 'COLAB_GPU' in os.environ:
        environment = 'COLAB'
        print("🤔 Running on Google Colab")
        if not allow_colab:
            raise Exception(colab_exception_message)
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        environment = 'KAGGLE'
        print("🤔 Running on Kaggle")
        if not allow_kaggle:
            raise Exception(kaggle_exception_message)
    elif allow_unrecognized:
        environment = 'UNRECOGNIZED'
        print("⚠️ Running in a unrecognized environment.")
    else:
        raise Exception(unrecognized_environment_exception_message)

    environment_initialized = True

def load_environment_variable(variable_name):
    if not environment_initialized:
        raise Exception("Environment not yet initialized. Run `deep_atlas.initialize_environment()` first.")
    if environment == 'VIRTUAL':
        # We shouldn't need to install packages in this environment, since Pipenv handles that for us
        from dotenv import load_dotenv
        load_dotenv()
        if variable_name not in os.environ:
            raise Exception(f"{variable_name} does not exist in os.environ")
        return os.environ[variable_name]
    elif environment == 'COLAB':
        from google.colab import userdata
        return userdata.get(variable_name)

def load_openai_api_key():
    return load_environment_variable('OPENAI_API_KEY')


def log_student_entry(entry={}):
    logspath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    logsfile = os.path.join(logspath, "logs.json")

    current_time = datetime.datetime.now().isoformat()
    notebook_filename = None
    if "__file__" in sys.modules["__main__"].__dict__:
        notebook_filename = sys.modules["__main__"].__file__
        if notebook_filename.endswith(".ipynb"):
            notebook_filename = os.path.splitext(
                os.path.basename(notebook_filename)
            )[0]
        else:
            notebook_filename = None
    if notebook_filename is None:
        ip = get_ipython()
        notebook_filename = ip.user_ns.get("__file__")
        if "__vsc_ipynb_file__" in ip.user_ns:
            notebook_filename = ip.user_ns["__vsc_ipynb_file__"]
        if notebook_filename is not None:
            notebook_filename = os.path.splitext(
                os.path.basename(notebook_filename)
            )[0]

    try:
        with open(logsfile, "r") as f:
            logs = json.load(f)
    except FileNotFoundError:
        logs = []

    if isinstance(entry, dict):
        logs.append(
            {
                "created_at": current_time,
                "notebook": notebook_filename,
                **entry,
            }
        )
    else:
        logs.append(
            {
                "created_at": current_time,
                "notebook": notebook_filename,
                "entry": entry,
            }
        )
    with open(logsfile, "w") as f:
        json.dump(logs, f, indent=2)


def log_start_time():
    log_student_entry({"event": "start"})
    print("🚀 Success! Get started...")


def log_stop_time():
    log_student_entry({"event": "stop"})
    print("🎉 Thanks! You're all done.")


def log_feedback(feedback):
    log_student_entry({"event": "feedback", "feedback": feedback})
    print("📝 Feedback logged!")


def log_assessment(responses):
    if any([response == FILL_THIS_IN for response in responses.values()]):
        print("❌ Please respond to all questions before submitting your assessment.")
        return
    log_student_entry({"event": "assessment", "responses": responses})
    print("📝 Assessment logged!")
