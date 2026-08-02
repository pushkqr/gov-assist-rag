import os, sys
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()
from core.utils import get_weaviate_client

wc = get_weaviate_client()
col = wc.collections.get("GovDocs")
count = col.aggregate.over_all(total_count=True).total_count
print(f"GovDocs object count: {count}")

# Also sample a few objects to see what fields look like
result = col.query.fetch_objects(limit=3, include_vector=False)
for obj in result.objects:
    p = obj.properties
    print(f"  - source_filename={p.get('source_filename')} | doc_number={p.get('doc_number')} | year={p.get('year')}")
wc.close()
