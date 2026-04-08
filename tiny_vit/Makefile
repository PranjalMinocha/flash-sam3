ASSETS_DIR := assets
BPE_FILE   := $(ASSETS_DIR)/bpe_simple_vocab_16e6.txt.gz
BPE_URL    := https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz

# Pass the signed URL for any single shard from the SA-1B download TSV, e.g.:
#   make data SA1B_URL="https://..."
SA1B_URL   ?=

.PHONY: all bpe data clean

all: bpe data

bpe: $(BPE_FILE)

$(BPE_FILE):
	mkdir -p $(ASSETS_DIR)
	wget -q --show-progress -O $(BPE_FILE) $(BPE_URL)

data:
ifndef SA1B_URL
	$(error SA1B_URL is not set. Pass a shard URL: make data SA1B_URL="<url>")
endif
	mkdir -p $(ASSETS_DIR)
	aria2c -x 16 -s 16 --dir $(ASSETS_DIR) --out sa_shard.tar "$(SA1B_URL)"
	tar -xf $(ASSETS_DIR)/sa_shard.tar -C $(ASSETS_DIR) --strip-components=1
	rm $(ASSETS_DIR)/sa_shard.tar

clean:
	rm -rf $(ASSETS_DIR)
