#!/usr/bin/env python3
"""
공공입찰 RAG LLM 비교 평가 — 동결 입력 manifest 빌더

questions_300.jsonl에서 '모델에 보여줄 것'만 뽑아 고정한다.
정답·판정·관리자 전용 근거는 전부 제거한다.
출력물은 9종 모델이 공통으로 사용하며, SHA-256으로 동일성을 증명한다.

사용법:
  python build_frozen_inputs.py <questions_300.jsonl> <출력디렉터리>
"""
import json, hashlib, sys, pathlib, datetime

# ─── 모델에 절대 노출하지 않는 필드 ────────────────────────────────
# 정답/판정/채점 기준 — 프롬프트에서 제거하고 누출 검사도 수행
EXCLUDE_ANSWER = [
    "expected_answer", "expected_action", "expected_status",
    "expected_overall_eligibility", "reasoning", "calculation",
    "must_include", "must_not_claim", "review", "hallucination_spans",
    "unresolved_issues", "assumptions", "actual_answer",
]
# 관리자 전용 근거 (권한 밖 자료)
EXCLUDE_RESTRICTED = ["private_audit_evidence"]
# 프롬프트에서는 제거하되 누출 검사 대상에서 제외하는 필드.
# answer_evidence_excerpt는 '정답 근거로 채택된 구간'인데, 그 구간 자체가
# 우리가 정상 제공하는 공고 인용문의 부분집합이다. 문자열이 겹치는 것은
# 설계상 당연하며 정답 누출이 아니다. 이 필드를 별도로 삽입하지는 않는다.
EXCLUDE_OVERLAP_BY_DESIGN = ["answer_evidence_excerpt"]
# 평가 운영용 메타데이터 (모델 입력에 들어가면 난이도 힌트가 됨)
EXCLUDE_META = [
    "difficulty", "split", "source_nature", "category", "condition_scope",
    "evidence_kinds", "master_pointers", "scenario_pointers",
    "scenario_evidence_ids", "synthetic_evidence_ids", "rag_ingest",
    "response_requirements",
]

# 공통 시스템 프롬프트 템플릿 (전 모델 동일)
SYSTEM_TEMPLATE = """당신은 공공입찰 문서 질의응답 도우미다.

출력은 두 줄로 한다. 첫 줄은 "답변:"으로 시작해 질문에 대한 결론과 이유를 쓴다. 근거를 사용한 문장 끝에는 그 근거의 번호를 적는다. 둘째 줄은 "근거:"로 시작해 사용한 근거 번호만 쉼표로 나열한다. 사용한 근거가 없으면 둘째 줄에 번호를 적지 않는다. 이 두 줄 외에 다른 제목이나 머리말은 쓰지 않는다.

규칙:
1. 아래 [근거]에 제시된 내용만 사용한다. 근거에 없는 내용은 추정하지 않는다.
2. 근거가 부족하면 첫 줄에 부족하다고 밝힌다.
3. 숫자·날짜·단위·부가세 포함 여부는 근거 표기를 그대로 따른다.
4. 열람 권한 밖의 자료를 요구하는 질문에는 답하지 않고 그 사실을 알린다.
5. 근거 원문을 그대로 옮겨 적지 않는다.

기준일: {decision_as_of}
질문자 역할: {role}
열람 가능 등급: {levels}"""

USER_TEMPLATE = """[근거]
{evidence}

[질문]
{question}"""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_evidence_block(q):
    """공고 근거 + 회사 근거를 번호 매겨 하나의 블록으로 만든다."""
    items = []
    for e in q.get("public_notice_evidence") or []:
        label = f"공고 {q.get('notice_name') or q.get('notice_id')} / {e.get('filename', '')}"
        items.append((label, e.get("quote") or "", e.get("source_id")))
    for e in q.get("company_evidence") or []:
        label = f"사내문서 {e.get('document_id')} (등급 {e.get('security_level')})"
        items.append((label, e.get("quote") or "", e.get("chunk_id")))

    lines, refs = [], []
    for i, (label, quote, src) in enumerate(items, 1):
        lines.append(f"[{i}] 출처: {label}\n{quote.strip()}")
        refs.append({"n": i, "label": label, "source_ref": src,
                     "chars": len(quote.strip())})
    if not lines:
        return "(제공된 근거 없음)", refs
    return "\n\n".join(lines), refs


def build(questions_path: str, out_dir: str):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = pathlib.Path(questions_path).read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    questions = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]

    rows, blocked, leak_check = [], [], []
    for q in questions:
        uc = q["user_context"]
        evidence, refs = build_evidence_block(q)

        system = SYSTEM_TEMPLATE.format(
            decision_as_of=q["decision_as_of"],
            role=uc["role"],
            levels=", ".join(uc["allowed_security_levels"]) or "(없음)",
        )
        user = USER_TEMPLATE.format(evidence=evidence, question=q["question"])

        # 서버 권한 필터가 차단하는 문항 — LLM 호출 대상에서 제외
        is_blocked = q["category"] == "access_control" and q["expected_status"] == "deny"

        row = {
            "question_id": q["question_id"],
            "tenant_id": q["tenant_id"],
            "company_id": q["company_id"],
            "blocked_by_acl": is_blocked,
            "system": system,
            "user": user,
            "evidence_refs": refs,
            "evidence_count": len(refs),
            "evidence_chars": sum(r["chars"] for r in refs),
            "prompt_chars": len(system) + len(user),
            "system_sha256": sha256(system),
            "user_sha256": sha256(user),
            "context_sha256": sha256(system + "\x00" + user),
        }
        rows.append(row)
        if is_blocked:
            blocked.append(q["question_id"])

        # 누출 검사: 제외 대상 필드의 문자열이 프롬프트에 섞이지 않았는지 확인
        blob = system + user
        for f in EXCLUDE_ANSWER + EXCLUDE_RESTRICTED:
            v = q.get(f)
            if isinstance(v, str) and len(v) > 40 and v[:40] in blob:
                leak_check.append((q["question_id"], f))
            if isinstance(v, list):
                for item in v:
                    s = item.get("quote") if isinstance(item, dict) else item
                    if isinstance(s, str) and len(s) > 40 and s[:40] in blob:
                        leak_check.append((q["question_id"], f))

    # 전체 동일성 증명용 집계 해시
    aggregate = sha256("\n".join(r["context_sha256"] for r in rows))

    inputs_path = out / "frozen_inputs.jsonl"
    with inputs_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "schema": "frozen-inputs/v2",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_file": pathlib.Path(questions_path).name,
        "source_file_sha256": source_sha,
        "question_count": len(rows),
        "generation_target_count": sum(1 for r in rows if not r["blocked_by_acl"]),
        "blocked_by_acl": blocked,
        "excluded_fields": {
            "answer_side": EXCLUDE_ANSWER,
            "restricted": EXCLUDE_RESTRICTED,
            "eval_metadata": EXCLUDE_META,
            "excluded_not_leak_tested": EXCLUDE_OVERLAP_BY_DESIGN,
        },
        "system_template_sha256": sha256(SYSTEM_TEMPLATE),
        "user_template_sha256": sha256(USER_TEMPLATE),
        "aggregate_context_sha256": aggregate,
        "leak_check": {"violations": len(leak_check), "detail": leak_check[:20]},
        "common_settings": {
            "quantization": "Q4_K_M",
            "context": 24576,
            "max_output_tokens": 512,
            "temperature": 0,
            "seed": 1234,
            "runtime": "llama.cpp",
            "build_commit": "0bec16e3880a148a7fc3887cdf71f81c547aff7a",
        },
        "stats": {
            "evidence_chars_total": sum(r["evidence_chars"] for r in rows),
            "prompt_chars_max": max(r["prompt_chars"] for r in rows),
            "prompt_chars_median": sorted(r["prompt_chars"] for r in rows)[len(rows) // 2],
            "max_prompt_question_id": max(rows, key=lambda r: r["prompt_chars"])["question_id"],
        },
    }
    (out / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"문항           : {len(rows)}")
    print(f"생성 대상      : {manifest['generation_target_count']}")
    print(f"ACL 차단       : {len(blocked)}  {blocked}")
    print(f"누출 검사 위반 : {len(leak_check)}")
    print(f"프롬프트 길이  : 중앙값 {manifest['stats']['prompt_chars_median']:,}자 / "
          f"최대 {manifest['stats']['prompt_chars_max']:,}자 "
          f"({manifest['stats']['max_prompt_question_id']})")
    print(f"집계 해시      : {aggregate[:16]}...")
    print(f"출력           : {inputs_path}")
    return 0 if not leak_check else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(build(sys.argv[1], sys.argv[2]))
