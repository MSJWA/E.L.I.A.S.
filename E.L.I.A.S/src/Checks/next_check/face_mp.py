import cv2, numpy as np, mediapipe as mp, time, os
from utils import FACE_DIR, save_embedding, load_embedding, list_face_users, log_event

mp_face = mp.solutions.face_mesh

# choose a stable set of landmark indices for embedding (eyes/nose/mouth/jaw region)
EMB_IDX = [1, 4, 5, 10, 33, 61, 199, 263, 291, 199, 168, 151, 130, 359, 386]

def _open_cam():
    # try multiple backends (MSMF first)
    for backend in (cv2.CAP_MSMF, cv2.CAP_ANY, cv2.CAP_DSHOW):
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap
                cap.release()
        except Exception:
            pass
    return None

def landmarks_from_frame(frame, face_mesh):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0]
    pts = [(p.x, p.y, p.z) for p in lm.landmark]
    return np.array(pts)  # shape (468,3)

def embedding_from_landmarks(pts):
    # normalize by face center + scale
    center = pts.mean(axis=0)
    ptsn = pts - center
    scale = np.linalg.norm(ptsn) + 1e-9
    ptsn = ptsn / scale
    # pick subset and flatten
    sel = ptsn[EMB_IDX].flatten()
    # also append global stats to be more robust
    stats = np.array([ptsn[:,0].mean(), ptsn[:,1].mean(), ptsn[:,2].mean()])
    return np.concatenate([sel, stats])

def register_face(user_id, samples=6):
    cap = _open_cam()
    if cap is None:
        print("No camera found")
        return False
    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as fm:
        embeds = []
        print("Look at camera. Capturing samples...")
        t0 = time.time()
        while len(embeds) < samples and (time.time() - t0) < 30:
            ret, frame = cap.read()
            if not ret:
                continue
            lm = landmarks_from_frame(frame, fm)
            if lm is not None:
                emb = embedding_from_landmarks(lm)
                embeds.append(emb)
                print(f"Captured {len(embeds)}/{samples}")
            cv2.putText(frame, f"Samples:{len(embeds)}/{samples}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
            cv2.imshow("Register - press Q to abort", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    if not embeds:
        print("No usable face captured.")
        return False
    mean_emb = np.mean(np.stack(embeds), axis=0)
    save_path = os.path.join(FACE_DIR, f"{user_id}.npy")
    save_embedding(save_path, mean_emb)
    log_event("face_register", f"user {user_id} registered", {"samples": len(embeds)})
    print(f"Saved face embedding: {save_path}")
    return True

def _load_all_embeddings():
    users = list_face_users()
    embs = []
    for u in users:
        path = os.path.join(FACE_DIR, f"{u}.npy")
        embs.append(load_embedding(path))
    return users, embs

def recognize_face_once(threshold=0.5):
    cap = _open_cam()
    if cap is None:
        return {"ok": False, "reason": "no_camera"}
    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as fm:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"ok": False, "reason": "no_frame"}
        lm = landmarks_from_frame(frame, fm)
        cap.release()
    if lm is None:
        return {"ok": False, "reason": "no_face"}
    emb = embedding_from_landmarks(lm)
    users, embs = _load_all_embeddings()
    if not users:
        return {"ok": False, "reason": "no_registered"}
    # cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity([emb], embs)[0]
    idx = int(np.argmax(sims))
    best_user = users[idx]
    best_score = float(sims[idx])
    if best_score >= threshold:
        return {"ok": True, "user": best_user, "score": best_score}
    else:
        return {"ok": False, "reason": "no_match", "best_score": best_score}
