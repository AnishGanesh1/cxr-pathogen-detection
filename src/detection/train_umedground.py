import torch
from torch.utils.data import DataLoader

from data.dataset import MockMRGDataset
from models.umedground import UMedGround
from utils.losses import BoxLoss, UncertaintyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MockMRGDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UMedGround().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

box_loss_fn = BoxLoss()
uncertainty_loss_fn = UncertaintyLoss()

for epoch in range(5):
    total_loss = 0

    for batch in loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bbox = batch["bbox"].to(device)

        optimizer.zero_grad()

        coarse_box, multi_boxes = model(image, input_ids, attention_mask)

        loss1 = box_loss_fn(coarse_box, bbox)
        loss2 = uncertainty_loss_fn(multi_boxes, bbox)

        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss}")
