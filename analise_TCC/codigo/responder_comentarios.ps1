param([string]$arquivo, [string]$json)
$resp = Get-Content -Raw -Encoding UTF8 $json | ConvertFrom-Json
function Norm([string]$s) { return (($s -replace '\s+', ' ').Trim()) }
$w = New-Object -ComObject Word.Application; $w.Visible = $false
try {
  $d = $w.Documents.Open($arquivo, $false, $false)
  $raizes = @()
  foreach ($c in $d.Comments) {
    $anc = $null; try { $anc = $c.Ancestor } catch { $anc = $null }
    if ($anc -eq $null) { $raizes += $c }
  }
  $feitas = 0; $falhas = @()
  foreach ($c in $raizes) {
    $t = Norm $c.Range.Text
    $achou = @($resp | Where-Object {
      if ($_.chave.StartsWith('=')) { $t -ceq (Norm $_.chave.Substring(1)) } else { $t.Contains((Norm $_.chave)) } })
    if ($achou.Count -ne 1) { $falhas += "$($achou.Count): $($t.Substring(0, [Math]::Min(60, $t.Length)))"; continue }
    [void]$c.Replies.Add($c.Scope, $achou[0].resposta)
    $feitas++
  }
  "raizes: $($raizes.Count)  respostas gravadas: $feitas  falhas: $($falhas.Count)"
  $falhas
  $d.Save(); $d.Close()
} finally { $w.Quit() }
