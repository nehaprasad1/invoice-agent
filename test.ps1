$body = @'
{"message":{"messageId":"t1","role":"ROLE_USER","parts":[{"mediaType":"application/vnd.ga5.invoice-claim-batch+json","data":{"batchId":"b1","policyRevision":"r1","packages":[{"packageId":"p1","text":"Invoice INV-100 from Acme Ltd for INR 1,200.00. PO matched and goods received. Within the autonomous limit. [PO-1] [GRN-1] [INV-100]"}]}}]},"configuration":{"returnImmediately":false}}
'@
$body | Out-File -Encoding ascii body.json
curl.exe -s -X POST https://invoice-agent-t0hj.onrender.com/a2a/message:send -H "Authorization: Bearer test" -H "A2A-Version: 1.0" -H "Content-Type: application/a2a+json" --data-binary "@body.json"