param(
    [string]$BaseUrl = "http://127.0.0.1:8002",
    [string]$Text = "This is a test sentence.",
    [string]$ImageUrl = "https://picsum.photos/300"
)

$ErrorActionPreference = "Stop"

function Invoke-EmbeddingRequest {
    param(
        [string]$Uri,
        [hashtable]$Body
    )

    $jsonBody = $Body | ConvertTo-Json -Compress
    Invoke-RestMethod -Method Post -Uri $Uri -ContentType "application/json" -Body $jsonBody
}

$embedUrl = "$BaseUrl/embed"

Write-Host "Testing text embedding at $embedUrl"
$textResponse = Invoke-EmbeddingRequest -Uri $embedUrl -Body @{
    datatype = "text"
    input = $Text
}
$textDim = @($textResponse.embedding).Count
Write-Host "Text request succeeded. time_cost=$($textResponse.time_cost)s, dim=$textDim"

Write-Host "Testing image embedding at $embedUrl"
$imageResponse = Invoke-EmbeddingRequest -Uri $embedUrl -Body @{
    datatype = "image"
    input = $ImageUrl
}
$imageDim = @($imageResponse.embedding).Count
Write-Host "Image request succeeded. time_cost=$($imageResponse.time_cost)s, dim=$imageDim"

Write-Host ""
Write-Host "Text preview:"
$textResponse | Format-List

Write-Host ""
Write-Host "Image preview:"
$imageResponse | Format-List
