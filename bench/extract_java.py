#!/usr/bin/env python3
"""Extract a multi-file Maven/Java project from a model's chat-completion JSON.

Models often skip the per-file path labels, so each file's path is derived from
its content: the xml block is pom.xml; each java block's `package` + public type
name (+ a Test/Tests/IT suffix -> test source) gives its path.

Usage: extract_java.py <out_json> <out_dir>
"""
import json, re, sys, os

content = json.load(open(sys.argv[1]))["choices"][0]["message"]["content"]
lines = content.split("\n"); blocks = []; i = 0
while i < len(lines):
    s = lines[i].lstrip()
    if s.startswith("```"):
        lang = s[3:].strip().lower(); body = []; i += 1
        while i < len(lines) and not lines[i].lstrip().startswith("```"):
            body.append(lines[i]); i += 1
        i += 1; blocks.append((lang, "\n".join(body)))
    else:
        i += 1
pkgre = re.compile(r'^\s*package\s+([\w.]+)\s*;', re.M)
typere = re.compile(r'(?:public\s+|final\s+|abstract\s+)*\b(class|interface|enum|record)\s+(\w+)')
outdir = sys.argv[2]
for lang, body in blocks:
    if 'xml' in lang or '<project' in body[:200]:
        path = "pom.xml"
    elif 'java' in lang or 'package ' in body[:200]:
        m = pkgre.search(body); pkg = m.group(1) if m else "com.example.bank"
        t = typere.search(body)
        if not t:
            continue
        name = t.group(2)
        kind = "test" if re.search(r'(Test|Tests|IT)$', name) else "main"
        path = f"src/{kind}/java/{pkg.replace('.', '/')}/{name}.java"
    else:
        continue
    full = os.path.join(outdir, path); os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, "w").write(body.rstrip() + "\n"); print("wrote", path)
