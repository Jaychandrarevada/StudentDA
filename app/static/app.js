const $ = (id) => document.getElementById(id);

$("trainBtn").onclick = async () => {
  $("trainOut").textContent = "Training in progress...";
  const res = await fetch('/api/train', { method: 'POST' });
  const json = await res.json();
  $("trainOut").textContent = JSON.stringify(json, null, 2);
};

$("metricsBtn").onclick = async () => {
  const res = await fetch('/api/metrics');
  const json = await res.json();
  $("metricsOut").textContent = JSON.stringify(json, null, 2);
};

$("predictForm").onsubmit = async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const payload = Object.fromEntries(f.entries());
  ["attendance_pct","internal_score","assignment_avg","quiz_avg","lms_logins_per_week","lms_hours_per_week","content_views","forum_posts","previous_gpa"]
    .forEach(k => payload[k] = Number(payload[k]));

  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const json = await res.json();
  $("predictOut").textContent = JSON.stringify(json, null, 2);
};
