pragma solidity >=0.6.11 <0.9.0;
interface IVerifierFixedInput {
    function verifyProof(
        uint[2] calldata a,
        uint[2][2] calldata b,
        uint[2] calldata c,
        uint[10] calldata input
    ) external view returns (bool r);
}
contract ProofRegistryV1 {
    // Scope enum (0 = NONE, 1 = RSU, 2 = GLOBAL)
    uint8 public constant SCOPE_RSU = 1;
    uint8 public constant SCOPE_GLOBAL = 2;
    struct ProofRecordV1 {
        uint8 scope;
        bytes32 runId;
        uint32 rsuId;
        uint32 roundIdx;
        bool verifiedOk;
        bytes32 proofSha256;
        bytes32 publicInputsSha256;
        bytes32 vkeySha256;
        bytes32 verifierSolSha256;
        bytes32 manifestSha256;
        address submitter;
        uint256 blockNumber;
        uint256 timestamp;
    }
    mapping(bytes32 => ProofRecordV1) public proofRecords;
    mapping(bytes32 => uint256) public runSubmitted;
    mapping(bytes32 => uint256) public runVerifiedOk;
    mapping(bytes32 => bool) public runFinalized;
    mapping(bytes32 => bytes32) public globalProofKey;
    IVerifierFixedInput public rsuVerifier;
    IVerifierFixedInput public globalVerifier;
    event ProofVerifiedV1(
        bytes32 indexed proofKey,
        uint8 indexed scope,
        bytes32 indexed runId,
        uint32 rsuId,
        uint32 roundIdx,
        bool verifiedOk,
        bytes32 proofSha256,
        bytes32 publicInputsSha256,
        bytes32 vkeySha256,
        bytes32 verifierSolSha256,
        bytes32 manifestSha256,
        address submitter
    );
    event RunFinalizedV1(
        bytes32 indexed runId,
        bytes32 globalProofKey,
        bool globalVerifiedOk,
        uint256 totalSubmitted,
        uint256 totalVerifiedOk
    );
    constructor(address rsuVerifierAddr, address globalVerifierAddr) {
        rsuVerifier = IVerifierFixedInput(rsuVerifierAddr);
        globalVerifier = IVerifierFixedInput(globalVerifierAddr);
    }
    function _proofKey(
        uint8 scope,
        bytes32 runId,
        uint32 rsuId,
        uint32 roundIdx,
        bytes32 proofSha256,
        bytes32 publicInputsSha256,
        bytes32 vkeySha256,
        bytes32 verifierSolSha256,
        bytes32 manifestSha256
    ) internal pure returns (bytes32) {
        return keccak256(
            abi.encodePacked(
                scope, runId, rsuId, roundIdx,
                proofSha256, publicInputsSha256,
                vkeySha256, verifierSolSha256, manifestSha256
            )
        );
    }
    function submitAndVerifyRSU_V1(
        bytes32 runId,
        uint32 rsuId,
        uint32 roundIdx,
        uint[2] calldata a,
        uint[2][2] calldata b,
        uint[2] calldata c,
        uint[10] calldata input,
        bytes32 proofSha256,
        bytes32 publicInputsSha256,
        bytes32 vkeySha256,
        bytes32 verifierSolSha256,
        bytes32 manifestSha256
    ) external returns (bytes32 proofKey, bool ok) {
        proofKey = _proofKey(
            SCOPE_RSU, runId, rsuId, roundIdx,
            proofSha256, publicInputsSha256, vkeySha256, verifierSolSha256, manifestSha256
        );
        require(proofRecords[proofKey].scope == 0, "already submitted");
        ok = rsuVerifier.verifyProof(a, b, c, input);
        proofRecords[proofKey] = ProofRecordV1(
            SCOPE_RSU, runId, rsuId, roundIdx, ok,
            proofSha256, publicInputsSha256, vkeySha256, verifierSolSha256, manifestSha256,
            msg.sender, block.number, block.timestamp
        );
        runSubmitted[runId] += 1;
        if (ok) {
            runVerifiedOk[runId] += 1;
        }
        emit ProofVerifiedV1(
            proofKey, SCOPE_RSU, runId, rsuId, roundIdx, ok,
            proofSha256, publicInputsSha256, vkeySha256, verifierSolSha256, manifestSha256,
            msg.sender
        );
        return (proofKey, ok);
    }
    function submitAndVerifyGLOBAL_V1(
        bytes32 runId,
        uint32 roundIdx,
        uint[2] calldata a,
        uint[2][2] calldata b,
        uint[2] calldata c,
        uint[10] calldata input,
        bytes32 proofSha256,
        bytes32 publicInputsSha256,
        bytes32 vkeySha256,
        bytes32 verifierSolSha256
    ) external returns (bytes32 proofKey, bool ok) {
        bytes32 manifestSha256 = bytes32(0);
        proofKey = _proofKey(
            SCOPE_GLOBAL, runId, 0, roundIdx,
            proofSha256, publicInputsSha256, vkeySha256, verifierSolSha256, manifestSha256
        );
        require(proofRecords[proofKey].scope == 0, "already submitted");
        ok = globalVerifier.verifyProof(a, b, c, input);
        proofRecords[proofKey] = ProofRecordV1(
            SCOPE_GLOBAL, runId, 0, roundIdx, ok,
            proofSha256, publicInputsSha256, vkeySha256, verifierSolSha256, manifestSha256,
            msg.sender, block.number, block.timestamp
        );
        globalProofKey[runId] = proofKey;
        runSubmitted[runId] += 1;
        if (ok) {
            runVerifiedOk[runId] += 1;
        }
        emit ProofVerifiedV1(
            proofKey, SCOPE_GLOBAL, runId, 0, roundIdx, ok,
            proofSha256, publicInputsSha256, vkeySha256, verifierSolSha256, manifestSha256,
            msg.sender
        );
        return (proofKey, ok);
    }
    function finalizeRunV1(bytes32 runId) external returns (bool globalOk) {
        require(!runFinalized[runId], "already finalized");
        bytes32 gkey = globalProofKey[runId];
        require(gkey != bytes32(0), "missing global proof");
        globalOk = proofRecords[gkey].verifiedOk;
        runFinalized[runId] = true;
        emit RunFinalizedV1(
            runId, gkey, globalOk, runSubmitted[runId], runVerifiedOk[runId]
        );
        return globalOk;
    }
}
