from setuptools import setup, find_packages

setup(
    name="llm-redteam-harness",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["openai"],
    entry_points={
        "console_scripts": [
            "llm-redteam=llm_red_teaming_script:main",
        ],
    },
    license="CC0-1.0",
    description="A reproducible harness for red-teaming gpt-oss-20b",
    author="Soumadeep Das",
)
