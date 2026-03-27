import os

with open('tail_output.txt', 'w', encoding='utf-8') as out:
    out.write("--- DASHBOARD ---\n")
    try:
        with open('dashboard.log', 'rb') as f:
            f.seek(-2000, os.SEEK_END)
            data = f.read()
            out.write(data.decode('utf-8', errors='ignore'))
    except Exception as e:
        out.write(f"Error: {e}\n")

    out.write("\n--- BOT ---\n")
    try:
        with open('bot.log', 'rb') as f:
            f.seek(-4000, os.SEEK_END)
            data = f.read()
            out.write(data.decode('utf-8', errors='ignore'))
    except Exception as e:
        out.write(f"Error: {e}\n")
