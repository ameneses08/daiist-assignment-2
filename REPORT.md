# Assignment 2 Report

- **Name**: Antonio Meneses Esclante
- **Student ID**: 19015
- **Email**: ameneses.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem, and the cost of being wrong

The decision: At a ripening center in Spain, bananas imported green from Costa Rica are ripened and packed for supermarkets. Currently, a quality inspector, hand by hand, inspects whether each hand can be sent to the supermarket or must be rejected. The decision the model supports is this one, and the inspector or the line acts on it. 

Positive class: Rotten. One case is one image, which shows one hand of bananas. A hand is truly defective if it contains at least one Rotten label. The model's score for each image is the highest rotten confidence it predicts. If the model says rotten and it's truly rotten, then the hand is correctly rejected and doesn't reach the supermarket. Nevertheless, if the model says rotten and the hand was truly good, then food is being thrown away. On the other hand, if the model says that the hand is good but it's really rotten, this one reaches the supermarket. Finally, if the model classifies the hand as good and it's truly good, then it correctly passes. 

Why the two mistakes cost different amounts: A false positive throws away a perfectly good hand, worth what the supermarket would have paid for it. A false negative puts a rotten hand on a box directly headed to the supermarket. In the case that the supermarket finds this one, it would most likely reject the whole box. Therefore, a miss tends to cost more than a false alarm, €7.50 against €0.85, roughly nine times more. In the end its worth throwing away some good hands in order to catch the rotten ones. 


## Dataset

Link: https://universe.roboflow.com/fruit-pnemw/fruit-jmtdq

Source: Roboflow Universe, workspace fruit-pnemw, project fruit-jmtdq, version 1 ("Banana Quality Check"). Used as published, not forked. Object detection, classes Fresh and Rotten, 2,000 images stretched to 640×640. One image represents one banana hand.

Why this dataset? This one was initially compared against an apples dataset, nevertheless it only had 71 images in the test set with watermarks. This banana dataset has a test split 5.6 times larger and its images are closer to an inspection camera.

## What the dataset's composition implies

| | Train | Val | Test |
|---|---|---|---|
| Images | 1,200 | 400 | 400 |
| Images containing Rotten | 592 | 195 | 198 |
| Fresh boxes | 734 | 233 | 243 |
| Rotten boxes | 3,582 | 1,112 | 1,129 |

| Box size | Fresh | Rotten |
|---|---|---|
| Median area (% of image) | 70.2% | 0.58% |
| Smaller than 32×32 px | 0% | 32% |

The two classes are labelled at different levels: Fresh boxes cover the whole bananas, when rotten boxes only cover certain patches. Rotten's five times more boxes is a labelling artefact, per image the classes are balanced. 

The sample is adequate per image: With 198 rotten test images.

The images are screenshots taken on one day in 2018, not photos from an inspection line.

Leakage, found in two steps: A 64-bit difference hash flagged 57 test images as near-identical to a train image. Checking 12 pairs by eye, 6 were the same photo with grain or a brightness change and 6 were different photos that only looked alike. Moreover, the names keep the augmentation and the original photo, for example rotated_by_60_Screen-Shot-2018-06-12-at-9_04_0…. The dataset was uploaded already augmented.

Grouping images by original photo: 362 of 400 test images (90.5%) are copies of photos that
also appear in train; only 38 come from photos never seen. I kept the dataset as published,
made decisions on the full test set, and report the 38 unseen images as a check.

What I can't conclude: The test set is half rotten while production is about 3%, so I
convert every cost to 3%. The 38 unseen images have only 8 good hands, too few to trust a
false-alarm rate, and the screenshots don't show how the model would do on a real line.

## Finetuning and baselines

Object detection with yolo11n, finetuned from COCO-pretrained weights. Detection rather than classification because the dataset is labelled with boxes and rot is local. The model has to find brown spots on a hand that may be mostly yellow. One decision per hand comes from the highest Rotten confidence in the image. I initially began using yolo11s, due to its capacity to identify small brown spots in the fruit. Nevertheless at about 2.5 minutes per epoch it required over two hours of training, I stopped and switched.  

Training: MacBook Pro, Apple M3 Pro GPU (MPS); Ultralytics 8.4.173, PyTorch 2.11.0.
Image size 640, 30 epochs (patience 10, all 30 ran), batch 16, seed 0, AdamW chosen
automatically, default augmentation. 72.7 minutes.

Results on the same test split (400 images, positive class Rotten).

| Model | mAP50 | mAP50-95 | Fresh AP50 | Rotten AP50 | Rotten hands caught | Good hands wrongly rejected |
|---|---|---|---|---|---|---|
| Finetuned yolo11n | **0.786** | **0.673** | 0.916 | 0.598 | **198 / 198** | 2 / 202 |
| COCO yolo11s, zero-shot ("banana" counted as Fresh) | 0.251 | 0.221 | 0.502 | 0.000 | 0 / 198 | 0 / 202 |
| Trivial: predicts nothing (= pass every hand, the majority class) | 0.000 | 0.000 | 0.000 | 0.000 | 0 / 198 | 0 / 202 |

What changed: Finetuning added one thing the business needs: a notion of rot. Zero-shot COCO already knows what a banana is, but it has no rot
class, so it catches no rotten hand, which in practice is the same as the trivial baseline. After finetuning, Fresh rises to 0.92 and Rotten from 0 to 0.60, and at the
level of the hand the model catches all 198 rotten hands with 2 false alarms.

Was the gap as large as expected? Partly, I did expect COCO to score zero on Rotten, and it did. Moreover, I did expect Rotten to stay below Fresh, considering that a third of Rotten boxes are tiny spots. At the end it did (0.60 vs 0.92; the model finds about half the individual spots). The surprise was the size of the gap per hand. There are two reasons behind this: Finding one spot is enough to flag a hand, and the dataset's rotten bananas are mostly obvious. I also checked whether leakage
inflated it: on the 38 test images from photos never seen in training, the model scored the same.



## Per-class evaluation and failure modes

| Class | Boxes | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|---|
| Fresh | 243 | 0.927 | 0.918 | 0.968 | 0.927 |
| Rotten | 1,129 | 0.675 | 0.491 | 0.604 | 0.419 |
| All (headline) | 1,372 | 0.801 | 0.704 | 0.786 | 0.673 |

Which class carries the headline and which one drags: Fresh carries it (AP50 0.97);
Rotten drags it down (0.60). That matters because Rotten is the positive class, the one
whose misses cost €7.50. The drop from AP50 to AP50-95 shows why: Fresh barely changes
(0.97 → 0.93) while Rotten halves (0.60 → 0.42). The model often finds a rotten spot but
places its box imprecisely, which is expected when a third of Rotten boxes are smaller
than 32×32 pixels.

What the model confuses with what (box-level confusion matrix, full test set).

| Predicted ↓ / True → | Fresh (243) | Rotten (1,129) | Background |
|---|---|---|---|
| Fresh | **218** (90%) | 87 (8%) | 3,282 |
| Rotten | 25 (10%) | **1,018** (90%) | 31,948 |
| Background (missed) | 0 | 24 (2%) | — |

90% of rotten boxes are found as Rotten, 8% are boxed but called Fresh, and only 2% are missed entirely. 90% of fresh boxes are found
as Fresh; 10% are called Rotten. So the real confusion is between the two classes on the same fruit, not missing fruit.

Where it breaks:

1. Fresh vs. Rotten on whole bananas: 87 rotten boxes called Fresh, 25 fresh called Rotten.
2. Small spots: boxed imprecisely (Rotten AP50-95 0.42). Per hand this rarely matters, since one spot of about 5.7 is enough to flag the hand.
3. Noise looks like rot: one of the two false alarms is a salt-and-pepper copy, whose dark dots resemble rot spots.
4. Not in the test set: subtle defects and harmless sugar spots, so expect more borderline cases and false alarms in production.


## Cost matrix and operating point

| Outcome | Cost per hand | Where it comes from |
|---|---|---|
| False positive (good hand rejected) | €0.85 | A hand is about 1 kg; supermarkets pay about €15–16.30 per 18.4 kg box of Costa Rican bananas (≈ €0.85/kg). A rejected ripe banana can hardly be resold. |
| False negative (rotten hand shipped) | €7.50 | If the supermarket finds it, it rejects the whole box (≈ €15). I assume it finds one in two: 50% × €15. |
| True positive / true negative | €0 | The correct outcome. |

Base rate: 3% of hands rotten. Bananas are graded before export and shipped green in
refrigerated containers, where spoilage is kept low (TIS cargo guide).

Sweep: For every line from 0.00 to 1.00, I applied the test-set recall and
false-alarm rate to 1,000 production hands (30 rotten, 970 good) and priced the errors (`reports/cost_curve.png`).

Operating point: 0.04. Per 1,000 hands: 30 of 30 rotten caught, 9.6 good hands rejected, €8.20, against €225 for passing everything (bootstrap range €0–20.40).
The line is low (strict) because a miss costs about nine times a false alarm.

Assumption that would most change the answer: the cost of a miss, specifically
that supermarkets find half of the rotten hands. 0.04 stays optimal while a miss costs
at least €4.25, which means a detection rate of about 28% or more. Below that, a more
relaxed line (0.52) becomes cheaper. The same happens if fewer than 2% of hands are rotten.


## What a person still has to check

Thresholds: Pass below 0.04, a person checks 0.04–0.80, reject from 0.80. The cheapest band on the test set was 0.04–0.69, but 0.69 sat just above a single false
alarm (score 0.685), so I raised the high line to 0.80 as a safety margin. The borderline rotten hands (scores 0.40–0.52) fall inside this band.

Workload: (50,000 hands a day, based on a mid-sized Spanish ripening centre of 18,000 t/year at about 1 kg per hand):

| | Value |
|---|---|
| Share sent to a person | 2.7% |
| Hands checked per day | 1,329 |
| Time per check | 5 s |
| Checking time per day | 1.9 h |
| People needed (8-hour shifts) | 0.23 (0.46 at 10 s per check) |

Is it realistic? Yes. It's under a quarter of one person's shift, so an existing line
worker can run the review station without a new hire. Assumptions: €14.30 per hour (a
warehouse inspector plus employer social security), so about €0.02 per check, and the
person decides correctly.

Against the alternatives (per day):

| Policy | Missed | Good thrown away | People | Cost / day |
|---|---|---|---|---|
| Do nothing | 1,500 | 0 | 0 | €11,250 |
| Person checks everything | 0 | 0 | 8.7 | €993 |
| Model decides alone (0.04) | 0 | 480 | 0 | €408 |
| Model + review (0.04–0.80) | 0 | 0 | 0.23 | €26 |

The model does most of the work; review removes the 480 good hands a day that the strict
line would throw away. The safety margin costs about €9 a day more than the cheapest band.


## Limitations & next steps

| Limitation | How I'd address it |
|---|---|
| **Leakage:** 90.5% of test images are copies of training photos; only 38 are truly unseen. | Re-split by original photo (from the file names) so no photo crosses splits, then retrain and re-evaluate. |
| **Not production data:** screenshots of obviously rotten bananas, with few subtle defects or sugar spots. | Collect and label images from a real ripening-centre camera, including borderline hands, and set the thresholds on those. |
| **Inconsistent labels:** Rotten is sometimes a spot, sometimes a whole banana, causing Fresh/Rotten swaps. | Relabel with one convention (rot marked as spots only), or label each hand as pass/fail for a classifier. |
| **Thresholds rest on a few images:** the best line ranges from 0.02 to 0.52 across bootstrap samples. | Test on more good hands, then run a pilot on the line and track the real review and false-alarm rates. |
| **Cost of a miss is assumed:** a 50% supermarket detection rate. | Get the centre's chargeback records to measure how often rotten hands are found and what each costs. |
| **Review assumptions:** 5 s per check, reviewers always right. | Time the review station and measure how often two inspectors agree. |
| **Training cut short:** yolo11n, 30 epochs, still improving. | After fixing the split, train yolo11s for longer and compare minimum cost, not just mAP. |

## Generative AI use disclosure

I used Claude (Anthropic) throughout, as the AI policy allows.

- **Code:** Claude aided me with the construction of the scripts. I ran them on
  my machine, reviewed them, checked their outputs, and reported errors back to fix them.
- **Explanations and options:** for each decision, Claude trade-offs, I made the final decision. 
- **My decisions:** the problem and framing, the dataset, the decision unit, how to
  handle leakage, switching to yolo11n, every cost figure, the base rate, the threshold,
  the review band, the daily volume, the time per check and the wage.
- **Findings:** I checked the flagged duplicates by eye and saw they were the same
  photos with grain or lighting changes. Claude then pointed out that the file names
  revealed the augmentations and original photos, which led to the 90% leakage finding.
- **This report:** Claude did final review pointing out anything that I might have missed,
    helped me with grammar and reading comprehension errors, as well as making sure that
    the report maintained consistency. 