findings = [
    {"id": 1, "severity": "critical", "title": "SQL Injection"},
    {"id": 2, "severity": "low", "title": "Missing Header"},
    {"id": 3, "severity": "critical", "title": "Remote Code Execution"},
    {"id": 4, "severity": "medium", "title": "Weak Configuration"}
]


def get_critical_findings(findings):
    critical_findings = []
    for finding in findings:
        if(finding["severity"] == "critical"):            critical_findings.append(finding)
    return critical_findings

#print(get_critical_findings(findings))

def get_findings_by_severity(findings, title):
    critical_findings = []
    for finding in findings:
        if(finding["title"] == title):
            critical_findings.append(finding)
    return critical_findings

print(get_findings_by_severity(findings,"SQL Injection"))