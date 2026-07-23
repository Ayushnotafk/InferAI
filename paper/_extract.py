from pathlib import Path

t = Path("paper/_generate_paper.py").read_text(encoding="utf-8")
start = t.find("PAPER = r'''") + len("PAPER = r'''")
end = t.find("'''", start)
paper = t[start:end]
out = Path("paper/InferAI_Research_Paper.md")
out.write_text(paper, encoding="utf-8")
print("words", len(paper.split()))
print("chars", len(paper))
print("wrote", out)
