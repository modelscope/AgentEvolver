import re
TEMPLATE = """# Execution Records Compression Task

You need to intelligently compress and summarize execution records in the following format:

## Input Format
Each record block contains:
- **[action]**: Executed action (code, API calls, commands, etc.)
- **[state]**: Environment feedback/state information

## Compression Requirements

### 1. Drastically Reduce Redundant Information
- Remove detailed technical implementation details, keep core functionality descriptions
- Delete repetitive error messages, debug outputs, verbose logs
- Compress long JSON responses, retain only key fields and results
- Simplify lengthy code snippets, replace with functional descriptions

### 2. Merge Similar Operations
- Combine repeated operations with identical functionality into single records
- Similar operations with different parameters can be generalized (e.g., "logged into multiple accounts")
- Preserve core logic and key differences between operations

### 3. Maintain Important Distinctions
- Different types of operations must remain separate (e.g., "email login" vs "system login")
- Key parameter differences must be preserved (e.g., different API endpoints, important configurations)
- Success/failure status must be clearly indicated

### 4. Preserve Execution Order
- **CRITICAL**: Maintain the chronological sequence of operations
- Number each compressed entry to reflect original execution order
- Do not reorder operations for grouping purposes

### 5. Handle Failed Operations
- For failed operations, include the **root cause** of the failure
- Omit raw error messages and stack traces
- Focus on actionable failure reasons (e.g., "authentication failed due to invalid credentials" instead of full HTTP error response)

## Output Format
Provide a brief analysis first, then the compressed results:

```
Analysis: [1-2 sentences explaining your compression strategy]

COMPRESSED_RESULTS_START
1. Operation - Status - Key info [- Cause if failed]
2. Operation - Status - Key info [- Cause if failed]
3. Operation - Status - Key info [- Cause if failed]
COMPRESSED_RESULTS_END
```

## Compression Example
**Original Record** (verbose):
```
## 1.
[action]
requests.post('https://api.example.com/login', json={'username': 'user1', 'password': 'wrongpass'})

[state]
HTTP 401 Unauthorized
{"error": "Authentication failed", "code": "INVALID_CREDENTIALS", "message": "The provided username or password is incorrect", "timestamp": "2024-01-15T10:30:45Z", "request_id": "req_12345"}

## 2.
[action]
requests.post('https://api.example.com/login', json={'username': 'user1', 'password': 'correctpass'})

[state]
HTTP 200 OK
{"status": "success", "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...", "user_id": 12345, "permissions": ["read", "write"], "expires_in": 3600}
```

**Compressed Output**:
```
Analysis: Prioritized unique operations, omitted redundant retries.

COMPRESSED_RESULTS_START
1. API Login - Failed - Username 'user1' - Invalid credentials
2. API Login - Success - JWT token obtained, user ID 12345
COMPRESSED_RESULTS_END
```

## Length Constraints
- **STRICT LIMIT**: Final output must not exceed 2048 characters
- Due to length constraints, you may need to omit some blocks

Now please compress the following records while maintaining their chronological order:

{{records}}
"""

def get_prompt_compress_states(records:str):
    return TEMPLATE.replace("{{records}}", records)

def parse_compressed_results(llm_output:str):
    """Extract the compressed results as a simple string."""
    pattern = r'COMPRESSED_RESULTS_START\s*\n(.*?)\nCOMPRESSED_RESULTS_END'
    match = re.search(pattern, llm_output, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return ""