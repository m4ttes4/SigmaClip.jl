

# main wrapper of Workspace for type stability
struct SigmaClipWorkspace{C <: AbstractVector, S <: Union{Nothing, AbstractVector}}
    buf::C 
    aux::S 
end

@inline workspace_buffer(ws::SigmaClipWorkspace) = ws.buf
@inline workspace_auxbuffer(ws::SigmaClipWorkspace) = ws.aux


"""
    SigmaClip.workspace_buffer(ws) -> AbstractVector

Return the main mutable scratch buffer used by SigmaClip for packed valid data.
Custom workspace types can participate in the allocation-free API by returning
a writable, 1-indexed `AbstractVector`.
"""
workspace_buffer(ws) = throw(
    ArgumentError(
        "unsupported workspace $(typeof(ws)); implement SigmaClip.workspace_buffer(::$(typeof(ws))) and SigmaClip.workspace_auxbuffer(::$(typeof(ws)))"
    )
)

"""
    SigmaClip.workspace_auxbuffer(ws) -> AbstractVector

Return the auxiliary mutable scratch buffer used by SigmaClip's specialised
`mad_std!` path and workspace-aware statistics. Custom workspace types may
return `nothing` when they do not provide auxiliary scratch space.
"""
workspace_auxbuffer(ws) = throw(
    ArgumentError(
        "unsupported workspace $(typeof(ws)); implement SigmaClip.workspace_buffer(::$(typeof(ws))) and SigmaClip.workspace_auxbuffer(::$(typeof(ws)))"
    )
)


function prepare_ws(data::AbstractArray{T}, spread::S, ::Nothing) where {T, S}
    
    buf = Vector{T}(undef, length(data))

    if need_aux(spread) #resolved al comptime
        aux = Vector{float(T)}(undef, length(data))
    else
        aux = nothing
    end

    return SigmaClipWorkspace(buf, aux)
end


function prepare_ws(data::AbstractArray{T}, spread::S, ws::WS) where {T, S, WS}
    #most of this checks should be resolved at comp-time
    n = length(data)

    buf = workspace_buffer(ws)
    aux = workspace_auxbuffer(ws)

    is_vec(buf)

    if eltype(buf) !== T 
        throw(ArgumentError("Provided buffer Type $(eltype(buf)) do not match data Type $T "))
    end

    if length(buf) < n
        throw(DimensionMismatch(
            "workspace buffer too short: length $(length(buf)) < required at least $n"
        ))
    end

    if need_aux(spread)
        isnothing(aux) && throw(ArgumentError("workspace aux buffer is required by this statistic but workspace_auxbuffer returned nothing"))
        if length(aux) < n
            throw(
                DimensionMismatch(
                    "workspace aux buffer too short: length $(length(buf)) < required at least $n"
                )
            )
        end
    end

    return ws
end


is_vec(::AbstractVector) = nothing
is_vec(_) = throw(ArgumentError("Workspace main buffer must be of type <: AbstractVector"))

