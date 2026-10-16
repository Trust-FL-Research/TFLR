import hashlib
import json
import time
import sys
import numpy as np

class Block:
    def __init__(self, index, event_data, previous_hash, signer):
        self.index = index
        self.timestamp = time.time()
        self.event_data = event_data
        self.previous_hash = previous_hash
        self.signer = signer
        self.hash, self.block_string = self.compute_hash()
        self.generation_time_ms = (time.time() - self.timestamp) * 1000
        self.size_bytes = sys.getsizeof(self.block_string)

    def compute_hash(self):
        block_string = json.dumps(
            {"index": self.index, "timestamp": self.timestamp, 
             "event": self.event_data, "previous_hash": self.previous_hash, "signer": self.signer},
            sort_keys=True,
        )
        return hashlib.sha256(block_string.encode()).hexdigest(), block_string

class TrustAwareBlockchain:
    def __init__(self, authorities, num_nodes):
        self.authorities = authorities
        self.total_latency_ms = 0.0
        self.total_size_bytes = 0
        self.chain = [self.create_genesis_block()]
        self.trust_scores = np.ones(num_nodes)

    def create_genesis_block(self):
        return Block(0, {"type": "Genesis"}, "0", "system")

    def add_block(self, event_data, signer):
        previous_hash = self.chain[-1].hash
        new_block = Block(len(self.chain), event_data, previous_hash, signer)
        self.chain.append(new_block)
        self.total_latency_ms += new_block.generation_time_ms
        self.total_size_bytes += new_block.size_bytes
        return new_block.hash

    def evaluate_trust(self, audited_losses):
        """
        ارزیابی بر اساس خطای محاسبه شده توسط سرور روی داده‌های پاک.
        تنظیم شده برای جلوگیری از مثبت کاذب (False Positive).
        """
        median_loss = np.median(audited_losses)
        mad = np.median(np.abs(audited_losses - median_loss))
        
        for i, loss in enumerate(audited_losses):
            deviation = (loss - median_loss) / (mad + 1e-5)
            
            # شرط ترکیبی: هم انحراف داشته باشد و هم خطای مطلق آن از 0.6 بیشتر باشد
            # این کار مانع از حذف گره‌های صادق در زمان افت شدید واریانس شبکه می‌شود
            if deviation > 2.0 and loss > 0.6:
                self.trust_scores[i] = 0.0  # طرد کامل مهاجمان
            else:
                self.trust_scores[i] = min(1.0, self.trust_scores[i] + 0.10) # بازپروری
                
        return self.trust_scores