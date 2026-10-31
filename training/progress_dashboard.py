"""Read-only local dashboard for the running training experiment."""
import json
from pathlib import Path
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1] / "artifacts/model_training/minicpm5_1b_hotpotqa"


def read_json(name):
    path = ROOT / name
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def status():
    log = ROOT / "training.log"
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace") if log.exists() else "")
    steps = re.findall(r"^\s*(\d+)\s+([\d.]+)", text, re.MULTILINE)
    step, loss = steps[-1] if steps else ("0", None)
    state = read_json("workflow_state.json")
    phase = state.get("phase", "waiting_for_training")
    names = {"waiting_for_training": "Training", "after": "Benchmarking trained model", "report": "Writing report", "complete": "Complete", "failed": "Failed", "training_incomplete": "Training stopped early"}
    predictions = ROOT / "predictions_after.jsonl"
    count = sum(1 for _ in predictions.open()) if predictions.exists() else 0
    return {"phase": names.get(phase, phase), "step": int(step), "loss": loss,
            "total_steps": 2500, "examples": min(5000, int(step) * 2), "evaluation": count,
            "baseline": read_json("metrics_baseline.json"), "after": read_json("metrics_after.json"),
            "error": state.get("error"), "log_updated": datetime.fromtimestamp(log.stat().st_mtime).astimezone().isoformat() if log.exists() else None,
            "report_ready": (ROOT / "experiment_report.md").exists()}


def examples():
    def records(name):
        path = ROOT / name
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    before = {r["id"]: r for r in records("predictions_baseline.jsonl")}
    after = {r["id"]: r for r in records("predictions_after.jsonl")}
    result = []
    for row in records("benchmark_validation.jsonl"):
        match = re.search(r"Question: (.*?)\n\nEvidence context:", row["prompt"], re.S)
        b, a = before.get(row["id"]), after.get(row["id"])
        if not b or not a:
            continue
        outcome = ("both_correct" if b["exact_match"] else "newly_correct") if a["exact_match"] else ("regressed" if b["exact_match"] else "both_incorrect")
        result.append({"id": row["id"], "question": match.group(1) if match else row["id"],
                       "answer": row["answer"], "evidence": row["supporting_facts"],
                       "type": row["type"], "before": b, "after": a, "outcome": outcome})
    return result


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model training · Live progress</title>
<style>
*{box-sizing:border-box}body{background:#10151d;color:#edf2f7;font:16px system-ui,sans-serif;margin:0;padding:36px 20px}main{max-width:880px;margin:auto}h1{font-size:32px;letter-spacing:-1px;margin:12px 0}p{color:#abb7c8;line-height:1.6}.tag{color:#7fe0b0;font-size:13px;font-weight:650;letter-spacing:1px}section{padding:25px;background:#18212d;border:1px solid #2b3849;border-radius:18px;margin:22px 0}.top{display:flex;justify-content:space-between;gap:20px;align-items:center}.big{font-size:38px;font-weight:700}.muted{color:#abb7c8;font-size:14px}progress{width:100%;height:22px;margin:22px 0;accent-color:#72dab2}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.value{font-size:28px;font-weight:650;margin:12px 0}small{color:#abb7c8}a{color:#8bc2ff}#error{color:#ffb5ad;white-space:pre-wrap}#result{color:#7fe0b0}#connection{margin-top:24px} @media(max-width:550px){.grid{grid-template-columns:1fr}.big{font-size:30px}h1{font-size:26px}}
button,select,input{font:inherit;background:#10151d;color:#edf2f7;border:1px solid #42566e;border-radius:8px;padding:10px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.controls input{flex:1;min-width:160px}pre.response{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.65 system-ui,sans-serif;color:#edf2f7}.answer{color:#7fe0b0;overflow-wrap:anywhere}#question{line-height:1.5;overflow-wrap:anywhere}.comparison{align-items:start}.comparison section{margin:8px 0}#example-count{margin:12px 0}.discussion li{margin:12px 0;line-height:1.6;color:#abb7c8}
</style><main><div class="tag">MODEL TRAINING · RESULTS</div><h1>MiniCPM5-1B experiment</h1><p>Training progress and the final comparison on the same 500 held-out questions.</p>
<p><a href="/learn" target="_blank">Watch the animated training lesson</a> · <a href="/guide" target="_blank">Read the interview study guide</a></p>
<section><div class="top"><div><div class="muted">Current stage</div><h2 id="phase">Connecting…</h2></div><div class="big" id="percent">—</div></div><progress id="bar" max="2500" value="0"></progress><div class="top"><span id="steps">Reading training log…</span><span id="examples"></span></div><p id="eval"></p><div class="muted" id="loss"></div><p id="error" role="alert"></p></section>
<div class="grid"><section><div class="muted">BEFORE TRAINING · 500 QUESTIONS</div><div class="value" id="before">—</div><small>Exact match · fully correct answers</small><p id="beforef1"></p></section><section><div class="muted">AFTER TRAINING · SAME 500 QUESTIONS</div><div class="value" id="after">Pending</div><small>Results appear after benchmarking finishes</small><p id="afterf1"></p><div id="result"></div></section></div>
<p id="updated"></p><p id="connection" aria-live="polite"></p><p id="report"></p>
<section id="dataset"><h2>What did we train on?</h2><p><a href="https://huggingface.co/datasets/hotpotqa/hotpot_qa" target="_blank">HotpotQA distractor</a>: English factual QA using Wikipedia passages. Each input contains a question and evidence passages; each target contains supporting evidence and the gold answer.</p><p id="training-counts"></p><div id="training-samples"></div><p class="muted">These are actual training examples, separate from the 500 evaluation questions below. Displayed targets are the exact shortened targets used for training. Dataset-derived material: <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank">CC BY-SA 4.0</a>; authors and sources are linked in the full report.</p><h3>Where did measured performance improve?</h3><p>Bridge questions connect facts across passages. Comparison questions compare entities, dates or quantities. These are question types, not subject-domain expertise ratings.</p><div id="type-results"></div><p class="muted">No separate medicine, coding, mathematics or other subject-specific benchmark was run.</p></section>
<section id="examples"><h2>Explore every evaluation question</h2><p>Original outputs are shown verbatim, including failures and cut-off answers. The gold answer was used for scoring, not supplied as a target during evaluation.</p><div class="controls"><label for="outcome">Show</label><select id="outcome"><option value="all">All 500 questions</option><option value="newly_correct">Newly correct after training</option><option value="regressed">Correct before, wrong after</option><option value="both_incorrect">Wrong in both runs</option><option value="both_correct">Correct in both runs</option></select><input id="search" aria-label="Search questions or IDs" placeholder="Search question or ID"></div><p class="muted" id="example-count" aria-live="polite">Loading question comparisons…</p><div class="controls"><button id="prev-question" type="button">Previous question</button><button id="next-question" type="button">Next question</button></div><h3 id="question"></h3><p class="muted" id="question-meta"></p><p class="answer"><strong>Gold answer: </strong><span id="gold"></span></p><div class="grid comparison"><section><h3>Before fine-tuning</h3><p id="score-before"></p><pre class="response" id="raw-before"></pre><p class="muted" id="parsed-before"></p></section><section><h3>After fine-tuning</h3><p id="score-after"></p><pre class="response" id="raw-after"></pre><p class="muted" id="parsed-after"></p></section></div><details><summary>Annotated supporting evidence</summary><p id="support"></p></details></section>
<section class="discussion"><h2>What should we conclude?</h2><p>The model improved on this specific context-based QA task. We have not established its performance ceiling or a general intelligence score.</p><ul><li><strong>Net gain:</strong> 120 previously wrong answers became correct; 34 previously correct answers regressed. That leaves 162/500 correct, up from 76/500.</li><li><strong>Remaining weakness:</strong> 338 answers still fail exact match. Token F1 of 48.57 shows partial answer overlap, not broad reliability.</li><li><strong>Output failures:</strong> empty parsed answers increased from 1 to 18. The 96-token output budget and evidence-first format need investigation; a causal diagnosis requires another controlled experiment.</li><li><strong>No demonstrated ceiling:</strong> one seed, one epoch and one set of LoRA settings do not establish that more learning is impossible.</li><li><strong>Next experiments:</strong> use a separate development set to test concise answer-first targets, output budget, evidence selection, data quality, learning rate, epochs and rank. Improvement is possible but not guaranteed.</li><li><strong>Preserve a final exam:</strong> after inspecting these questions, treat them as an explored benchmark. Select future settings on development data and use a fresh untouched final evaluation for new claims.</li></ul><p><a href="/report" target="_blank">Read the full report and limitations</a> · <a href="https://arxiv.org/abs/1809.09600" target="_blank">About the HotpotQA task</a></p></section>
</main>
<script>
const el=id=>document.getElementById(id);const fmt=n=>Number(n).toFixed(2);
async function refresh(){try{const response=await fetch('/status',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);const s=await response.json();
el('phase').textContent=s.phase;el('percent').textContent=fmt(s.step/s.total_steps*100)+'%';el('bar').value=s.step;
el('steps').textContent=s.step.toLocaleString()+' / '+s.total_steps.toLocaleString()+' training updates';el('examples').textContent=s.examples.toLocaleString()+' / 5,000 examples';
el('loss').textContent=s.loss?'Latest training loss: '+s.loss:'';el('eval').textContent=s.evaluation?'After-training benchmark: '+s.evaluation+' / 500 questions':'Next: benchmark the trained model on the saved 500 questions.';el('error').textContent=s.error||'';
if(s.baseline.count){el('before').textContent=fmt(s.baseline.exact_match)+'%';el('beforef1').textContent='Token F1: '+fmt(s.baseline.token_f1)+' / 100';}
if(s.after.count){el('after').textContent=fmt(s.after.exact_match)+'%';el('afterf1').textContent='Token F1: '+fmt(s.after.token_f1)+' / 100';const delta=s.after.exact_match-s.baseline.exact_match;el('result').textContent='Exact match change: '+(delta>=0?'+':'')+fmt(delta)+' percentage points';}
if(s.after.by_type&&s.baseline.by_type){el('type-results').replaceChildren();for(const type of ['bridge','comparison']){const b=s.baseline.by_type[type],a=s.after.by_type[type];if(!a||!b)continue;const p=document.createElement('p');p.textContent=`${type==='bridge'?'Bridge':'Comparison'} (${a.count} questions): exact match ${fmt(b.exact_match)}% → ${fmt(a.exact_match)}% (+${fmt(a.exact_match-b.exact_match)} points); token F1 ${fmt(b.token_f1)} → ${fmt(a.token_f1)}.`;el('type-results').appendChild(p);}}
el('updated').textContent=s.log_updated?'Training log last changed: '+new Date(s.log_updated).toLocaleString():'';
el('connection').textContent='Connected · checked '+new Date().toLocaleTimeString();if(s.report_ready){el('report').innerHTML='<a href="/report" target="_blank">Read the completed experiment report</a>';}
}catch(e){el('connection').textContent='Connection lost. The displayed values may be out of date. Retry in 3 seconds.';}}refresh();setInterval(refresh,3000);
let allExamples=[],filtered=[],exampleIndex=0;
const outcomes={newly_correct:'Newly correct',regressed:'Regressed',both_incorrect:'Incorrect in both runs',both_correct:'Correct in both runs'};
function showExample(){const row=filtered[exampleIndex];el('example-count').textContent=filtered.length?`${exampleIndex+1} / ${filtered.length} matching questions · ${allExamples.length} total`:'No matching questions';el('prev-question').disabled=!row||exampleIndex===0;el('next-question').disabled=!row||exampleIndex===filtered.length-1;
for(const id of ['question','question-meta','gold','score-before','score-after','raw-before','raw-after','parsed-before','parsed-after','support'])el(id).textContent='';if(!row)return;
el('question').textContent=row.question;el('question-meta').textContent=`${row.id} · ${row.type} · ${outcomes[row.outcome]}`;el('gold').textContent=row.answer;el('support').textContent=row.evidence;
for(const label of ['before','after']){const value=row[label];el('score-'+label).textContent=`Exact match: ${value.exact_match?'Correct':'Incorrect'} · Token F1: ${fmt(value.token_f1*100)} / 100`;el('raw-'+label).textContent=value.raw_completion||'(No generated response)';el('parsed-'+label).textContent='Scored answer: '+(value.prediction||'(empty)');}}
function filterExamples(){const term=el('search').value.trim().toLowerCase(),outcome=el('outcome').value;filtered=allExamples.filter(r=>(outcome==='all'||r.outcome===outcome)&&(!term||(r.question+' '+r.id).toLowerCase().includes(term)));exampleIndex=0;showExample();}
el('outcome').onchange=filterExamples;el('search').oninput=filterExamples;el('prev-question').onclick=()=>{if(exampleIndex>0){exampleIndex--;showExample();}};el('next-question').onclick=()=>{if(exampleIndex<filtered.length-1){exampleIndex++;showExample();}};
async function loadExamples(){try{const response=await fetch('/examples',{cache:'no-store'});if(!response.ok)throw Error();allExamples=await response.json();filterExamples();if(allExamples.length<500)setTimeout(loadExamples,15000);}catch(e){el('example-count').textContent='Question comparisons could not be loaded. Refresh to retry.';}}loadExamples();
async function loadTraining(){try{const response=await fetch('/training-data',{cache:'no-store'});if(!response.ok)throw Error();const data=await response.json();el('training-counts').textContent=`5,000 training examples: ${data.question_type_counts.bridge.toLocaleString()} bridge and ${data.question_type_counts.comparison.toLocaleString()} comparison questions. Difficulty labels: ${data.difficulty_counts.easy.toLocaleString()} easy, ${data.difficulty_counts.medium.toLocaleString()} medium, ${data.difficulty_counts.hard.toLocaleString()} hard.`;for(const row of data.samples){const h=document.createElement('h3'),p=document.createElement('p'),d=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');h.textContent=row.question;p.textContent=`Gold answer: ${row.answer} · ${row.type} · ${row.level}`;summary.textContent='See the actual teaching target';pre.className='response';pre.textContent=row.teaching_target;d.append(summary,pre);el('training-samples').append(h,p,d);}}catch(e){el('training-counts').textContent='Training sample summary is unavailable.';}}loadTraining();
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/":
            body, kind = HTML.encode(), "text/html; charset=utf-8"
        elif route == "/status":
            body, kind = json.dumps(status()).encode(), "application/json"
        elif route == "/examples":
            body, kind = json.dumps(examples()).encode(), "application/json"
        elif route == "/training-data":
            body, kind = json.dumps(read_json("training_data_summary.json")).encode(), "application/json"
        elif route == "/report" and (ROOT / "experiment_report.md").exists():
            body, kind = (ROOT / "experiment_report.md").read_bytes(), "text/plain; charset=utf-8"
        elif route == "/learn":
            body, kind = (ROOT / "fine-tuning-lesson-preview.html").read_bytes(), "text/html; charset=utf-8"
        elif route == "/guide":
            body, kind = (Path(__file__).parent / "INTERVIEW_GUIDE.md").read_bytes(), "text/plain; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("Live progress: http://127.0.0.1:8767", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8767), Handler).serve_forever()
