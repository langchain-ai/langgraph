package pregel

import (
	"context"
	"time"

	"github.com/google/uuid"
)

func EmptyCheckpoint() (*Checkpoint, error) {
	uid, err := uuid.NewV6()
	if err != nil {
		return nil, err
	}
	return &Checkpoint{
		Version:         1,
		ID:              uid.String(),
		Timestamp:       time.Now().Format(time.RFC3339),
		ChannelValues:   map[string]interface{}{},
		ChannelVersions: map[string]int64{},
		VersionsSeen:    map[string]interface{}{},
		PendingSends:    []Send{},
	}, nil
}

type Checkpointer interface {
	PutWrites(ctx context.Context, checkpoint Checkpoint, writes []Write) error
}
