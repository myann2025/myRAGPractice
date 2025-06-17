from src.agent import agent

if __name__ == "__main__":
    while (prompt := input("Enter a prompt (q to quit): ")) != "q":
        result = agent.query(prompt)
        print(result)
