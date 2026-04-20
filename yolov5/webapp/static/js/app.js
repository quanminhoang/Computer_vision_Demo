const form = document.getElementById('uploadForm');
const input = document.getElementById('imageInput');
const runBtn = document.getElementById('runBtn');
const statusText = document.getElementById('statusText');
const originalImg = document.getElementById('originalImg');
const processedImg = document.getElementById('processedImg');
const cropList = document.getElementById('cropList');
const countBadge = document.getElementById('countBadge');

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.style.color = isError ? '#b00020' : '#596067';
}

function renderCrops(crops) {
  cropList.innerHTML = '';
  countBadge.textContent = String(crops.length);

  if (!crops.length) {
    cropList.innerHTML = '<p>Không có biển số nào được phát hiện.</p>';
    return;
  }

  for (const item of crops) {
    console.log('CROP ITEM:', item);

    const card = document.createElement('article');
    card.className = 'crop-card';

    const cropSrc = item.crop_url 
      ? item.crop_url + '?t=' + Date.now()
      : '';

    const wrapper = document.createElement('div');
    wrapper.style = "display:flex; gap:10px; margin-bottom:10px; justify-content:center;";

    // ảnh gốc
    const originalDiv = document.createElement('div');
    originalDiv.style = "text-align:center;";
    originalDiv.innerHTML = `
      <small style="display:block; color:#666">Ảnh gốc</small>
      <img src="${cropSrc}" 
           style="width:140px; border:1px solid #ddd; border-radius:4px;" />
    `;

    // ảnh xử lý từ backend
    const enhancedDiv = document.createElement('div');
    enhancedDiv.style = "text-align:center;";
    const enhancedImg = document.createElement('img');
    enhancedImg.style = "width:140px; border:2px solid #007bff; border-radius:4px;";
    
    enhancedDiv.innerHTML = `<small style="display:block; color:#007bff">Đã xử lý</small>`;
    enhancedDiv.appendChild(enhancedImg);

    const enhancedSrc = item.enhanced_url
      ? item.enhanced_url + '?t=' + Date.now()
      : cropSrc;
    enhancedImg.src = enhancedSrc;

    wrapper.appendChild(originalDiv);
    wrapper.appendChild(enhancedDiv);

    card.appendChild(wrapper);

    // meta info
    const meta = document.createElement('div');
    meta.innerHTML = `
      <div class="meta"><strong>Biển số:</strong> 
        <span style="color:#d32f2f; font-weight:bold;">${item.text || '---'}</span>
      </div>
      <div class="meta">Độ tin cậy: ${item.confidence ? (item.confidence * 100).toFixed(1) : 0}%</div>
      <div class="meta">Nguồn OCR: ${item.ocr_source || '---'}</div>
      <div class="meta">Độ tin OCR: ${item.ocr_confidence ? (item.ocr_confidence * 100).toFixed(1) : 0}%</div>
      <div class="meta">Độ mờ: ${item.blur_score ?? '---'}</div>
      <div class="meta">Loại biển: ${item.plate_type || 'unknown'}</div>
      <div class="meta">Trạng thái: Đã xử lý</div>
    `;

    // 👉 bảng so sánh OCR
if (item.candidates && item.candidates.length) {
  const table = document.createElement('table');
  table.style = "width:100%; margin-top:10px; font-size:13px; border-collapse:collapse;";

  table.innerHTML = `
    <tr style="background:#f5f5f5">
      <th style="border:1px solid #ddd; padding:4px">Method</th>
      <th style="border:1px solid #ddd; padding:4px">Text</th>
      <th style="border:1px solid #ddd; padding:4px">Conf</th>
    </tr>
  `;

  item.candidates.forEach(c => {
    const row = document.createElement('tr');

    row.innerHTML = `
      <td style="border:1px solid #ddd; padding:4px">${c.type}</td>
      <td style="border:1px solid #ddd; padding:4px">${c.text || '---'}</td>
      <td style="border:1px solid #ddd; padding:4px">${(c.confidence * 100).toFixed(1)}%</td>
    `;

    if (c.type === item.ocr_source) {
      row.style.background = "#e3f2fd"; // highlight cái được chọn
    }

    table.appendChild(row);
  });

  card.appendChild(table);
}


    card.appendChild(meta);
    cropList.appendChild(card);
  }
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  if (!input.files.length) {
    setStatus('Ban chua chon anh.', true);
    return;
  }

  runBtn.disabled = true;
  setStatus('Dang nhan dien...');

  try {
    const body = new FormData();
    body.append('image', input.files[0]);

    const res = await fetch('/api/recognize', { method: 'POST', body });
    const data = await res.json();

    console.log('API RESPONSE:', data);

    if (!res.ok || !data.ok) {
      throw new Error(data.error || 'Loi khong xac dinh');
    }

    originalImg.src = data.original_url + '?t=' + Date.now();
    processedImg.src = data.processed_url + '?t=' + Date.now();

    renderCrops(data.crops || []);

    setStatus(`Hoan tat. Da phat hien ${data.count} bien so.`);
  } catch (err) {
    console.error(err);
    setStatus(`Loi: ${err.message}`, true);
  } finally {
    runBtn.disabled = false;
  }
});