"""Turn an agent message into a readable card body: a bold headline (first
sentence) followed by bullets, with action labels bolded and ' / ' lists
split into sub-bullets. Display only; the message text itself is unchanged."""
import re

LABELS = ("이연 대신 대체 조치:", "대체 조치:", "검증 통과:", "검증 실패:", "제안:", "조치:", "권고:")
SENT_SPLIT = re.compile(r"(?<=\.)\s+(?=[가-힣A-Za-z(])")


def _item(sentence):
    s = sentence.strip().rstrip(".")
    label, rest = "", s
    for lb in LABELS:
        if lb in s:
            pre, rest = s.split(lb, 1)
            label = (pre.strip() + " " if pre.strip() else "") + f"<span class='lbl'>{lb[:-1]}</span> "
            rest = rest.strip()
            break
    if " / " in rest:
        subs = "".join(f"<li>{x.strip().rstrip('.')}</li>" for x in rest.split(" / "))
        return f"<li>{label}<ul>{subs}</ul></li>"
    return f"<li>{label}{rest}</li>"


def format_body(text):
    text = re.sub(r"^\[[^\]]*\]\s*", "", text.strip())   # header already shows time and type
    sentences = [p.strip() for line in text.split("\n") for p in SENT_SPLIT.split(line.strip()) if p.strip()]
    if not sentences:
        return ""
    title, rest = sentences[0], sentences[1:]
    for lb in LABELS:                                     # a label inside the headline moves to the bullets
        if lb in title:
            head, tail = title.split(lb, 1)
            title, rest = head.strip(), [lb + tail] + rest
            break
    body = f"<div class='title'>{title.rstrip('.')}</div>"
    if rest:
        body += "<ul>" + "".join(_item(x) for x in rest) + "</ul>"
    return body
