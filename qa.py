import json

with open("qa_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def guess_category(item):
    q = item.get("question", "").lower()
    tags = [t.lower() for t in item.get("tags", [])]
    if "wudu" in q or "ablution" in q or "tayamum" in q or "ghusl" in q or "purity" in q or "taharah" in q or "ritual purity" in tags:
        return "Ablution and Purification"
    if "khoms" in q or "khums" in q or "zakat" in q or "zakah" in q:
        return "Zakat and Khoms Obligations"
    if "marriage" in q or "nikah" in q or "copulate" in q or "family" in q:
        return "Marriage and Family"
    if "makeup" in q or "kohl" in q or "eyeliner" in q or "modesty" in tags or "awra" in q:
        return "Modesty and Clothing"
    if "fast" in q or "ramadan" in q or "sawm" in q:
        return "Fasting and Rituals"
    if "death" in q or "burial" in q or "takfeen" in q or "shrouding" in q:
        return "Death and Burial"
    if "adhan" in q or "iqamah" in q or "prayer" in q or "salat" in q:
        return "Prayer and Ritual Acts"
    if "finance" in q or "loan" in q or "interest" in q or "bank" in q or "riba" in q:
        return "Finance and Islamic Law"
    if "well" in q or "water" in q:
        return "General Purity"
    return "General"

for item in data:
    if "category" not in item or not item["category"]:
        item["category"] = guess_category(item)

with open("qa_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)