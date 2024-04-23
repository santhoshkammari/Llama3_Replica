"""
python -m experiment_scripts.python_fire --name hai --age adf

"""
import fire

def hello(name,age):
	return "Hello " + name + age

if __name__ == '__main__':
	fire.Fire(hello)
